"""Structured-output runner + model-run observability (spec §71, §72, §74).

run_structured(): prompt -> provider chain -> parse -> validate -> (repair once) -> next
provider -> typed result or StructuredOutputError. Every attempt is recorded as a
`model_runs` row with prompt id/version, schema version, latency, tokens, cost, status.

Model-run rows are COLLECTED in a context variable and persisted after the caller's
transaction has finished (commit or rollback), in their own transaction. That way a failed
job still leaves a trace of exactly which model call failed and why.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select, update

from ..db.models import InterviewSession, ModelRun
from ..db.session import db_session
from ..errors import Unavailable
from . import pricing
from .guard import DATA_RULES
from .prompts import PromptSpec
from .provider import CompletionResult, ProviderError
from .routing import resolve
from .schema_text import schema_text

log = logging.getLogger("ii.ai")
T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, prompt_id: str, message: str):
        super().__init__(f"{prompt_id}: {message}")
        self.prompt_id = prompt_id


class BudgetExceeded(Unavailable):
    code = "ai_budget"


@dataclass
class RunContext:
    user_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    # When set, a call is refused once today's total spend reaches this many USD. Only the
    # entry points that START new spend (session creation, analysis) set it; a live turn of
    # an interview already in progress is protected by the per-session cap instead, so a
    # candidate is never cut off mid-answer by the global switch.
    daily_budget_usd: Optional[float] = None


@dataclass
class _RunRecord:
    data: Dict[str, Any] = field(default_factory=dict)


_collector: contextvars.ContextVar[Optional[List[_RunRecord]]] = contextvars.ContextVar("ii_runs", default=None)


@contextmanager
def collect_runs() -> Iterator[List[_RunRecord]]:
    token = _collector.set([])
    runs: List[_RunRecord] = []
    try:
        yield _collector.get()  # type: ignore[misc]
    finally:
        runs = _collector.get() or []
        _collector.reset(token)
        if runs:
            persist_runs(runs)


def persist_runs(runs: List[_RunRecord]) -> None:
    try:
        with db_session() as db:
            per_session: Dict[uuid.UUID, float] = {}
            for r in runs:
                db.add(ModelRun(**r.data))
                sid = r.data.get("session_id")
                if sid:
                    per_session[sid] = per_session.get(sid, 0.0) + float(r.data.get("cost_usd") or 0)
            for sid, c in per_session.items():
                if c:
                    db.execute(update(InterviewSession).where(InterviewSession.id == sid)
                               .values(cost_usd=InterviewSession.cost_usd + c))
    except Exception:  # noqa: BLE001 - observability must never break the product path
        log.exception("failed to persist %d model runs", len(runs))


def _record(**data: Any) -> None:
    rec = _RunRecord(data=data)
    bucket = _collector.get()
    if bucket is not None:
        bucket.append(rec)
    else:
        persist_runs([rec])


# USD per 1M characters for text-to-speech models (char-priced, not token-priced).
TTS_PRICE_PER_M_CHARS = {"tts-1": 15.0, "tts-1-hd": 30.0, "gpt-4o-mini-tts": 12.0}


def record_media_run(*, stage: str, provider: str, model: str, user_id: Optional[uuid.UUID], latency_ms: int,
                     status: str, chars: int = 0, audio_bytes: int = 0, error: str = "") -> None:
    """Speech calls are logged like every other model call (spec §71, §95). TTS is priced per
    character; STT cost depends on audio duration, which is not decoded here, so it is logged
    with the `unpriced_model` flag and the byte size rather than a made-up number."""
    cost = 0.0
    validation: Dict[str, Any] = {"audio_bytes": audio_bytes} if audio_bytes else {}
    if stage == "tts" and model in TTS_PRICE_PER_M_CHARS:
        cost = round(chars / 1e6 * TTS_PRICE_PER_M_CHARS[model], 6)
        validation["chars"] = chars
    elif provider != "simulated":
        validation["unpriced_model"] = model
    _record(user_id=user_id, session_id=None, stage=stage, provider=provider, model=model, prompt_id=stage,
            prompt_version="", schema_version="", input_tokens=0, output_tokens=0, cost_usd=cost,
            latency_ms=latency_ms, status=status, attempt=1, error=error[:2000], validation=validation)


def pending_cost(session_id: Optional[uuid.UUID]) -> float:
    bucket = _collector.get() or []
    return sum(float(r.data.get("cost_usd") or 0) for r in bucket
               if session_id is None or r.data.get("session_id") == session_id)


# ---- budgets -------------------------------------------------------------------------
_daily_cache: Dict[str, Any] = {"ts": 0.0, "day": "", "value": 0.0}


def spend_today_usd() -> float:
    now = time.time()
    day = datetime.now(timezone.utc).date().isoformat()
    if _daily_cache["day"] == day and now - _daily_cache["ts"] < 60:
        return float(_daily_cache["value"]) + pending_cost(None)
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    with db_session() as db:
        v = db.execute(select(func.coalesce(func.sum(ModelRun.cost_usd), 0)).where(ModelRun.created_at >= start)).scalar()
    _daily_cache.update(ts=now, day=day, value=float(v or 0))
    return float(v or 0) + pending_cost(None)


def reset_budget_cache() -> None:
    _daily_cache.update(ts=0.0, day="", value=0.0)


# ---- parsing -----------------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json_object(text: str) -> Any:
    t = (text or "").strip()
    t = _FENCE.sub("", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return json.loads(t[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", t, 0)


def _system(spec: PromptSpec, schema: Optional[Type[BaseModel]], extra_system: str) -> str:
    parts = [spec.system.strip(), DATA_RULES]
    if extra_system:
        parts.append(extra_system.strip())
    if schema is not None:
        parts.append(
            "OUTPUT FORMAT: respond with ONE JSON object only (no markdown, no prose) matching this shape. "
            "Use null or [] when you do not know; never invent facts to fill a field.\n" + schema_text(schema)
        )
    return "\n\n".join(parts)


def _log_attempt(spec: PromptSpec, ctx: RunContext, res: Optional[CompletionResult], provider: str, model: str,
                 status: str, attempt: int, latency_ms: int, error: str = "", validation: Optional[dict] = None):
    in_t = res.input_tokens if res else 0
    out_t = res.output_tokens if res else 0
    used_model = (res.model if res else model) or model
    cost, priced = pricing.cost_usd(used_model, in_t, out_t)
    v = dict(validation or {})
    if not priced and provider != "simulated":
        v["unpriced_model"] = used_model
    _record(user_id=ctx.user_id, session_id=ctx.session_id, stage=spec.stage, provider=provider,
            model=used_model, prompt_id=spec.id, prompt_version=spec.version,
            schema_version=spec.schema_version, input_tokens=in_t, output_tokens=out_t, cost_usd=cost,
            latency_ms=latency_ms, status=status, attempt=attempt, error=error[:2000], validation=v)


def _check_budget(ctx: RunContext) -> None:
    if ctx.daily_budget_usd and spend_today_usd() >= ctx.daily_budget_usd:
        raise BudgetExceeded("Interview Intelligence has reached today's capacity. Please try again tomorrow.")


def run_structured(spec: PromptSpec, user_content: str, schema: Type[T], ctx: RunContext, *,
                   sim_input: Optional[dict] = None, extra_system: str = "",
                   validate: Optional[Callable[[T], List[str]]] = None) -> T:
    """Return a validated `schema` instance or raise StructuredOutputError.

    `validate` may return semantic problems (list of strings); they trigger the same repair
    path as a schema error. `sim_input` is only read by the offline SimulatedProvider."""
    _check_budget(ctx)
    system = _system(spec, schema, extra_system)
    base_messages = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
    last_problem = "no provider attempted"
    for provider, model in resolve(spec.id, spec.route):
        messages = list(base_messages)
        for attempt in (1, 2):  # 2 = repair attempt on the same provider
            t0 = time.perf_counter()
            res: Optional[CompletionResult] = None
            try:
                res = provider.complete(messages, model=model, temperature=spec.temperature,
                                        max_tokens=spec.max_tokens, json_mode=True,
                                        timeout_s=spec.timeout_s,
                                        meta={"prompt_id": spec.id, "sim_input": sim_input, "attempt": attempt})
            except ProviderError as e:
                _log_attempt(spec, ctx, None, provider.name, model, "timeout" if e.timeout else "error", attempt,
                             int((time.perf_counter() - t0) * 1000), str(e))
                last_problem = str(e)
                break  # next provider
            latency = res.latency_ms or int((time.perf_counter() - t0) * 1000)
            problems: List[str] = []
            obj: Optional[T] = None
            try:
                obj = schema.model_validate(parse_json_object(res.text))
                if validate is not None:
                    problems = [p for p in (validate(obj) or []) if p]
            except json.JSONDecodeError as e:
                problems = [f"Output was not valid JSON ({e.msg})."]
            except ValidationError as e:
                problems = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:12]]
            if not problems and obj is not None:
                _log_attempt(spec, ctx, res, provider.name, model, "ok" if attempt == 1 else "repaired", attempt, latency)
                return obj
            _log_attempt(spec, ctx, res, provider.name, model, "invalid_output", attempt, latency,
                         error="; ".join(problems)[:2000], validation={"problems": problems[:12]})
            last_problem = "; ".join(problems)
            if attempt == 1:
                messages = base_messages + [
                    {"role": "assistant", "content": (res.text or "")[:6000]},
                    {"role": "user", "content": "Your previous output was rejected:\n- " + "\n- ".join(problems[:12])
                     + "\nReturn the corrected JSON object only."},
                ]
    raise StructuredOutputError(spec.id, last_problem)


def run_text(spec: PromptSpec, user_content: str, ctx: RunContext, *, sim_input: Optional[dict] = None,
             extra_system: str = "") -> str:
    """Free-text generation (the interviewer's spoken line). Returns '' if every provider failed —
    callers MUST have a deterministic fallback."""
    _check_budget(ctx)
    messages = [{"role": "system", "content": _system(spec, None, extra_system)},
                {"role": "user", "content": user_content}]
    for provider, model in resolve(spec.id, spec.route):
        t0 = time.perf_counter()
        try:
            res = provider.complete(messages, model=model, temperature=spec.temperature, max_tokens=spec.max_tokens,
                                    json_mode=False, timeout_s=spec.timeout_s,
                                    meta={"prompt_id": spec.id, "sim_input": sim_input, "attempt": 1})
        except ProviderError as e:
            _log_attempt(spec, ctx, None, provider.name, model, "timeout" if e.timeout else "error", 1,
                         int((time.perf_counter() - t0) * 1000), str(e))
            continue
        _log_attempt(spec, ctx, res, provider.name, model, "ok", 1, res.latency_ms or int((time.perf_counter() - t0) * 1000))
        if (res.text or "").strip():
            return res.text.strip()
    return ""
