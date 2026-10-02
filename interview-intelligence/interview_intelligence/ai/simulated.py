"""SimulatedProvider — a deterministic, offline stand-in for every AI stage.

Purpose: the full pipeline (upload -> analysis -> blueprint -> live interview -> evidence ->
evaluation -> feedback -> report) runs in CI and on a laptop without API keys. Its outputs
are plausible but SIMPLE heuristics; they exist to exercise schemas, guards, state machines
and failure paths — NOT to judge quality. Real assessment quality is measured with a real
model through `qa/run_golden.py`.

Enabled only when II_ENV is test/dev or II_ALLOW_SIMULATION=true (never silently in prod).
"""

from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

from .provider import CompletionResult, ProviderError

ACTION_VERBS = r"(led|managed|built|grew|increased|reduced|launched|delivered|drove|designed|owned|created|improved|" \
               r"negotiated|implemented|developed|analy[sz]ed|headed|scaled|cut|saved|won|spearheaded|automated)"
MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}"


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _num(s: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else None


def _slots(text: str) -> Dict[str, float]:
    t = text.lower().replace(",", "")
    out: Dict[str, float] = {}
    m = re.search(r"team of (\d+)|(\d+)[- ](member|person|people)\b|(\d+) (direct )?reports", t)
    if m:
        out["team_size"] = float(next(g for g in m.groups() if g and g.isdigit()))
    m = re.search(r"(grew|increas\w*|growth|improv\w*|boost\w*|rais\w*)[^.%]{0,40}?(\d+(?:\.\d+)?)\s?%", t)
    if m:
        out["growth_pct"] = float(m.group(2))
    m = re.search(r"(reduc\w*|sav\w*|cut\w*|lower\w*)[^.%]{0,40}?(\d+(?:\.\d+)?)\s?%", t)
    if m:
        out["cost_saving_pct"] = float(m.group(2))
    return out


# --------------------------------------------------------------------------------------- stages
def cv_parser(si: dict) -> dict:
    text = si.get("text", "")
    lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
    exp, claims = [], []
    for l in lines:
        m = re.search(rf"({MONTHS}|\d{{4}})\s*[-–to]+\s*({MONTHS}|present|current|\d{{4}})", l, re.IGNORECASE)
        if m and len(exp) < 6:
            head = l[:m.start()].strip(" ,|-–")
            parts = [p.strip() for p in re.split(r",| at | \| ", head) if p.strip()]
            exp.append({"id": f"X{len(exp) + 1}", "title": parts[0] if parts else head[:60],
                        "organization": parts[1] if len(parts) > 1 else "", "start": m.group(1), "end": m.group(3),
                        "is_current": m.group(3).lower() in ("present", "current")})
            continue
        if re.search(ACTION_VERBS, l, re.IGNORECASE) and len(l.split()) >= 5 and len(claims) < 12:
            sl = _slots(l)
            claims.append({
                "id": f"C{len(claims) + 1}", "text": l[:240],
                "type": "leadership" if re.search(r"\b(led|managed|headed|team)\b", l, re.I) else
                        ("impact" if re.search(r"\d", l) else "ownership"),
                "quantified": bool(re.search(r"\d", l)),
                "metrics": [{"value": _num(x), "unit": "%", "direction": "increase", "raw": x}
                            for x in re.findall(r"\d+(?:\.\d+)?\s?%", l)][:3],
                "slots": sl,
                "ownership_language": "team" if re.search(r"\b(we|our team|helped|assisted|part of)\b", l, re.I) else "personal",
                "vague": not re.search(r"\d", l) and bool(re.search(r"significant|various|several|many", l, re.I)),
                "needs_verification": True,
                "verification_priority": 1 if (sl or re.search(r"\b(led|managed)\b", l, re.I)) else 2,
            })
    skills_line = next((l for l in lines if l.lower().startswith("skills")), "")
    skills = [s.strip() for s in re.split(r"[,;|]", skills_line.split(":", 1)[-1]) if s.strip()][:15]
    years = re.search(r"(\d+)\+?\s+years", text, re.I)
    yrs = float(years.group(1)) if years else None
    sen = "unknown" if yrs is None else ("entry" if yrs < 2 else "mid" if yrs < 6 else "senior")
    missing = [s for s, rx in (("education", r"educat|university|college|degree|mba|b\.?tech"),
                               ("experience dates", MONTHS)) if not re.search(rx, text, re.I)]
    return {"headline": lines[0][:80] if lines else "", "current_title": exp[0]["title"] if exp else "",
            "total_experience_years": yrs, "experience_basis": "stated" if yrs else "unknown",
            "seniority_estimate": sen, "experience": exp, "claims": claims,
            "skills": {"technical": skills, "tools": [], "domain": [], "soft": [], "languages": []},
            "parse_quality": {"confidence": "medium", "missing_sections": missing, "notes": []}}


REQ_RX = re.compile(r"(experience|knowledge|skill|ability|proficien|degree|understanding|familiar|must|required|"
                    r"preferred|bachelor|mba|graduate|years)", re.I)


def jd_parser(si: dict) -> dict:
    text = si.get("text", "")
    lines = [l.strip(" -•*\t") for l in text.splitlines() if l.strip()]
    title = lines[0][:90] if lines else ""
    company = ""
    m = re.search(r"company\s*:\s*(.+)", text, re.I)
    if m:
        company = m.group(1).strip()[:80]
    reqs, resps = [], []
    for l in lines[1:]:
        if len(l.split()) < 3:
            continue
        if REQ_RX.search(l):
            imp = "must" if re.search(r"must|required|minimum", l, re.I) else (
                "nice" if re.search(r"plus|bonus|nice to have", l, re.I) else "should")
            reqs.append({"id": f"R{len(reqs) + 1}", "text": l[:240], "category": "functional_skill", "importance": imp,
                         "explicit": True})
        elif len(resps) < 12:
            resps.append({"id": f"RS{len(resps) + 1}", "text": l[:240], "kind": "major"})
    yrs = re.search(r"(\d+)\+?\s*(?:-\s*\d+\s*)?years", text, re.I)
    level = "unknown"
    if re.search(r"\bintern\b", text, re.I):
        level = "intern"
    elif yrs:
        y = int(yrs.group(1))
        level = "entry" if y < 2 else "mid" if y < 6 else "senior"
    words = re.findall(r"\b[A-Z][A-Za-z&+]{2,}(?:\s[A-Z][A-Za-z&+]{2,})?\b", text)
    kw = []
    for w in words:
        if w.lower() not in {x["term"].lower() for x in kw} and w.lower() not in ("the", "we", "you", "our", "this"):
            kw.append({"term": w, "category": "", "weight": "medium"})
    facts = [l for l in lines if re.search(r"\b(we are|our company|founded|leading|largest)\b", l, re.I)][:3]
    return {"identity": {"title": title, "company": company, "role_name": title},
            "seniority": {"level": level, "confidence": "medium" if level != "unknown" else "low",
                          "basis": yrs.group(0) if yrs else ""},
            "summary": " ".join(lines[1:3])[:300], "requirements": reqs, "responsibilities": resps,
            "keywords": kw[:20], "implied_competencies": [], "company_facts_in_jd": facts,
            "quality": {"specificity": "medium", "contradictions": [], "ambiguities": [], "notes": []}}


def role_classifier(si: dict) -> dict:
    fam = si.get("fallback") or "other"
    return {"primary_family": fam, "candidates": [{"family_id": fam, "confidence": 0.7}], "industry": "",
            "rationale": "simulated: keyword classification"}


def competency_mapper(si: dict) -> dict:
    from ..role_taxonomy.library import library
    from ..textutil import token_set
    lib = library()
    fam = lib.family(si.get("family") or "other")
    role, cand = si.get("role") or {}, si.get("candidate") or {}
    reqs = role.get("requirements", [])
    out = []
    for cid, imp in fam.defaults.items():
        c = lib.competency(cid)
        ctoks = token_set(f"{c.name} {c.definition} {' '.join(c.sub)}")
        rids = [r["id"] for r in reqs if len(token_set(r.get("text", "")) & ctoks) >= 2][:3]
        cl = [x["id"] for x in cand.get("claims", []) if len(token_set(x.get("text", "")) & ctoks) >= 2][:3]
        strong = any(x.get("quantified") for x in cand.get("claims", []) if x["id"] in cl)
        out.append({"competency_id": cid, "name": c.name, "importance": imp, "requirement_ids": rids,
                    "cv_claim_ids": cl, "cv_strength": "strong" if strong else ("partial" if cl else "none"),
                    "expected_depth": f"Applies {c.name.lower()} independently at the role's level.",
                    "sub_areas": fam.functional_areas[:2] if c.category in ("functional", "technical") else []})
    kws = role.get("keywords", [])
    if kws:
        k = kws[0]["term"]
        out.append({"competency_id": "rs:" + re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_"), "name": k,
                    "parent": "technical_depth" if fam.technical else "functional_knowledge", "importance": "medium",
                    "definition": f"Role-specific knowledge of {k}.",
                    "requirement_ids": [reqs[0]["id"]] if reqs else [], "sub_areas": [k]})
    return {"competencies": out}


def rubric_builder(si: dict) -> dict:
    from ..role_taxonomy.library import library
    lib = library()
    out = []
    for m in si.get("model", []):
        c = lib.competency(m["competency_id"]) or lib.competency(m.get("parent") or "") or lib.competency("functional_knowledge")
        out.append({"competency_id": m["competency_id"],
                    "what_good_looks_like": m.get("expected_depth") or c.definition,
                    "strong_signals": c.strong or ["specific, owned and reasoned evidence", "measurable outcome"],
                    "weak_signals": c.weak or ["generic statements", "no personal contribution"],
                    "red_flags": c.red, "expected_structure": c.structure, "must_not_infer": []})
    return {"rubrics": out}


TEMPLATES = {
    "functional": "In this role, how would you apply {topic}? Walk me through a concrete situation where it mattered.",
    "technical": "Explain how you would approach {topic} for a system this role owns, including the main trade-offs.",
    "behavioral": "Tell me about a time you had to show {topic} under real pressure. What did you personally do?",
    "situational": "Imagine your first month in this role: two urgent priorities clash and both involve {topic}. What do you do?",
    "case": "A business in this space sees margins fall by about 10% while volumes grow. How would you diagnose it, given {topic}?",
    "company": "Given what you know about this company, where would you focus first in this role, and why, considering {topic}?",
    "motivation": "What draws you to work on {topic} in this role specifically?",
}


def question_generator(si: dict) -> dict:
    section, n = si.get("section"), int(si.get("n", 2))
    comps = si.get("competencies") or []
    qs = []
    if section == "cv":
        for cl in (si.get("claims") or [])[:n]:
            qs.append({"text": f"You mention: \"{cl.get('text', '')[:140]}\". What exactly was your role, and how did "
                               f"you measure the result?",
                       "archetype_id": "cv_claim_verification", "competency_ids": [comps[0]["competency_id"]] if comps else [],
                       "question_type": "cv_deep_dive", "difficulty": si.get("difficulty", 3),
                       "intent": "Test whether the claim holds and what the candidate owned.",
                       "expected_evidence": ["personal actions", "baseline and result", "attribution"],
                       "strong_signals": ["clear personal decisions"], "weak_signals": ["'we' throughout"],
                       "probe_tree": ["What exactly did you own?", "What was the baseline?", "How much was attributable to you?"],
                       "claim_ids": [cl.get("id")], "why_this_question": "High-priority CV claim."})
    for i in range(len(qs), n):
        if not comps:
            break
        c = comps[i % len(comps)]
        topic = (c.get("sub_areas") or [c.get("name") or c["competency_id"]])[0]
        tmpl = TEMPLATES.get(section, TEMPLATES["functional"])
        qtype = {"cv": "cv_deep_dive", "company": "company"}.get(section, section)
        qs.append({"text": tmpl.format(topic=str(topic).lower()), "archetype_id": "", "competency_ids": [c["competency_id"]],
                   "question_type": qtype if qtype in ("behavioral", "situational", "functional", "technical", "case",
                                                      "cv_deep_dive", "motivation", "company") else "functional",
                   "difficulty": si.get("difficulty", 3), "intent": f"Assess {c.get('name')} in context.",
                   "expected_evidence": ["concrete example", "reasoning", "outcome"],
                   "strong_signals": ["specific and reasoned"], "weak_signals": ["generic"],
                   "probe_tree": ["Can you make that concrete?", "Why that approach?", "What was the result?"],
                   "why_this_question": "Covers a planned competency."})
    return {"questions": qs}


def blueprint_qa(si: dict) -> dict:
    return {"passed": True, "issues": [], "notes": ["simulated QA"]}


def turn_analyzer(si: dict) -> dict:
    ans = si.get("answer", "")
    words = len(ans.split())
    quantified = bool(re.search(r"\d", ans))
    reasoning = bool(re.search(r"\b(because|so that|therefore|which meant|the reason)\b", ans, re.I))
    owner = bool(re.search(r"\b(i|my)\b", ans, re.I))
    gaps, focus = [], "none"
    if not quantified:
        gaps.append("measurable outcome or baseline")
    if not owner:
        gaps.append("personal contribution vs the team's")
    if not reasoning:
        gaps.append("reasoning behind the choice")
    if gaps:
        focus = "quantification" if not quantified else ("ownership" if not owner else "reasoning")
    quality = "strong" if (words >= 60 and quantified and reasoning and owner) else ("weak" if words < 25 else "adequate")
    claims = []
    sl = _slots(ans)
    rel = (si.get("item") or {}).get("claim_ids") or [""]
    for k, v in sl.items():
        claims.append({"text": ans[:160], "relates_to": rel[0] if rel else "", "slot": k, "value": v,
                       "high_impact": True, "supported_in_answer": quantified})
    return {"intent": "answer", "addresses_question": "fully" if words > 40 else "partially", "answer_quality": quality,
            "specificity": 2 if quantified else 1, "ownership_clarity": 2 if owner else 0, "reasoning_present": reasoning,
            "quantified": quantified, "observed_signals": [], "gaps": gaps, "probe_focus": focus,
            "new_claims": claims, "summary": " ".join(ans.split()[:20])}


def interviewer(si: dict) -> str:
    return si.get("fallback") or "Could you tell me more?"


def evidence_extractor(si: dict) -> dict:
    text = si.get("candidate_text", "")
    allowed = si.get("allowed") or ["communication"]
    sents = [s for s in _sentences(text) if len(s.split()) >= 6][:5]
    items = []
    probed = bool(si.get("probed"))
    for i, s in enumerate(sents):
        if re.search(r"prefer not to answer|rather not answer|ignore (all )?previous", s, re.I):
            continue  # refusals and instructions to the system are not evidence
        negative = bool(re.search(r"not sure|don't know|no idea|can't remember|i guess", s, re.I))
        positive = bool(re.search(r"\d|because|i decided|i led|i built|i chose|trade-?off", s, re.I))
        # after a follow-up, a generic team-level reply is a tested gap, not an untested competency
        if probed and not positive and not re.search(r"\bi\b", s, re.I):
            negative = True
        items.append({"competency_id": allowed[i % len(allowed)], "type": "action",
                      "polarity": "negative" if negative else ("positive" if positive else "neutral"),
                      "strength": "strong" if (positive and re.search(r"\d", s) and re.search(r"\bi\b", s, re.I)) else "moderate",
                      "quote": " ".join(s.split()[:40]), "interpretation": "Simulated interpretation.",
                      "ownership": "personal" if re.search(r"\bi\b", s, re.I) else "team", "confidence": 0.6})
    dims = {d: {"applicable": True, "rating": 3, "note": "simulated"} for d in si.get("dims", [])}
    return {"claim": sents[0][:200] if sents else "", "items": items, "dimensions": dims,
            "what_worked": ["Gave a concrete example."] if items else [],
            "what_was_missing": ["A measurable outcome."] if not re.search(r"\d", text) else [],
            "interviewer_was_looking_for": "Specific, owned, measurable evidence.",
            "better_answer_direction": "Lead with your decision, then the result and how you measured it."}


def competency_evaluator(si: dict) -> dict:
    items = si.get("items", [])
    nn = [i for i in items if i["polarity"] != "neutral"]
    if not nn:
        return {"competency_id": si["competency_id"], "evidence_state": "not_sufficiently_tested", "score": None,
                "rationale": "Not enough evidence.", "evidence_refs": []}
    score = 5
    for i in nn:
        w = 2 if i["strength"] == "strong" else 1
        score += w if i["polarity"] == "positive" else -w
    score = max(1, min(9, score))
    st = "strong" if score >= 7 else ("moderate" if score >= 5 else "weak")
    return {"competency_id": si["competency_id"], "evidence_state": st, "score": score,
            "rationale": f"Simulated: {len(nn)} non-neutral evidence items.",
            "evidence_refs": [i["ref"] for i in nn], "strengths": [], "gaps": [] if score >= 7 else ["More specifics."]}


def assessment_qa(si: dict) -> dict:
    return {"findings": []}


def feedback_writer(si: dict) -> dict:
    assessments = si.get("assessments", [])
    exchanges = si.get("exchanges", [])
    strong = [a for a in assessments if a["evidence_state"] == "strong" and a.get("evidence_refs")]
    weak = [a for a in assessments if a["evidence_state"] in ("weak", "moderate") and a.get("evidence_refs")]
    untested = [a for a in assessments if a["evidence_state"] == "not_sufficiently_tested"]

    def xref(comp_id):
        for ex in exchanges:
            if any(e["competency"] == comp_id for e in ex.get("evidence", [])):
                q = next((e["quote"] for e in ex["evidence"] if e["competency"] == comp_id), "")
                return ex["ref"], q
        return (exchanges[0]["ref"], "") if exchanges else ("", "")

    devs = []
    for a in weak[:3]:
        x, quote = xref(a["competency_id"])
        devs.append({"category": "ownership" if a["competency_id"] == "ownership" else
                     ("communication" if a["competency_id"] == "communication" else "functional_depth"),
                     "severity": "medium", "title": f"Deepen evidence for {a['name'].lower()}",
                     "competency_ids": [a["competency_id"]],
                     "observed_problem": f"In your answer on this topic you described what happened but did not make your own "
                                         f"decisions and their measurable result explicit enough for {a['name'].lower()}.",
                     "example_refs": [x] if x else [], "example_quote": quote,
                     "why_it_matters": "Interviewers for this role look for owned, measurable evidence.",
                     "what_to_do": "State your decision in the first sentence, then the result with its baseline and timeframe.",
                     "practice": "Re-answer three questions on this competency aloud, each in under two minutes with a number."})
    return {
        "executive_assessment": (f"Across this interview you showed clear evidence in {len(strong)} competencies and partial "
                                 f"evidence in {len(weak)}. {len(untested)} competencies were not sufficiently tested, so "
                                 f"they are listed as missing evidence rather than weaknesses. This is a simulated summary."),
        "strengths": [{"title": f"Evidence of {a['name'].lower()}", "competency_ids": [a["competency_id"]],
                       "evidence_refs": a["evidence_refs"][:2], "why_it_matters": "Directly relevant to the role."}
                      for a in strong[:3]],
        "development_areas": devs,
        "interviewer_learned": {
            "strong_signals": [{"text": f"Demonstrated {a['name'].lower()}", "refs": a["evidence_refs"][:1]} for a in strong[:2]],
            "weak_signals": [{"text": f"Limited depth on {a['name'].lower()}", "refs": a["evidence_refs"][:1]} for a in weak[:2]],
            "unproven_claims": [], "missing_evidence": [{"text": f"{a['name']} was not tested", "refs": []} for a in untested[:3]],
            "potential_concerns": []},
        "next_questions": [{"question": f"Tell me about a time you demonstrated {a['name'].lower()}.",
                            "gap": f"{a['name']} was not sufficiently tested."} for a in untested[:3]],
        "preparation_plan": {"headline": "Your next practice session",
                             "items": [{"action": d["title"], "count": 3, "competency_ids": d["competency_ids"],
                                        "linked_development": [i], "how": d["practice"]} for i, d in enumerate(devs)],
                             "topics_to_revise": [], "reattempt_competencies": [a["competency_id"] for a in weak[:3]]},
    }


def feedback_qa(si: dict) -> dict:
    return {"development_areas": [{"index": i, "passed": True} for i in range(int(si.get("n", 0)))],
            "executive_assessment_ok": True}


HANDLERS: Dict[str, Callable[[dict], object]] = {
    "cv_parser": cv_parser, "jd_parser": jd_parser, "role_classifier": role_classifier,
    "competency_mapper": competency_mapper, "rubric_builder": rubric_builder, "question_generator": question_generator,
    "blueprint_qa": blueprint_qa, "turn_analyzer": turn_analyzer, "interviewer": interviewer,
    "evidence_extractor": evidence_extractor, "competency_evaluator": competency_evaluator,
    "assessment_qa": assessment_qa, "feedback_writer": feedback_writer, "feedback_qa": feedback_qa,
}


class SimulatedProvider:
    name = "simulated"

    def __init__(self, overrides: Optional[Dict[str, Callable[[dict, list], object]]] = None):
        self.overrides = overrides or {}
        self.calls: List[str] = []

    def complete(self, messages, *, model, temperature, max_tokens, json_mode, timeout_s, meta=None):
        meta = meta or {}
        pid = meta.get("prompt_id", "")
        self.calls.append(pid)
        si = meta.get("sim_input") or {}
        if pid in self.overrides:
            out = self.overrides[pid](si, messages)
        elif pid in HANDLERS:
            out = HANDLERS[pid](si)
        else:
            raise ProviderError(f"simulated provider has no handler for {pid}", retryable=False)
        if isinstance(out, Exception):
            raise out
        text = out if isinstance(out, str) else json.dumps(out)
        return CompletionResult(text=text, provider="simulated", model="simulated",
                                input_tokens=sum(len(m["content"]) for m in messages) // 4, output_tokens=len(text) // 4)

    def stream(self, messages, *, model, temperature, max_tokens, timeout_s, meta=None):
        yield self.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens, json_mode=False,
                            timeout_s=timeout_s, meta=meta).text

    def transcribe(self, audio, *, filename, mime, model, timeout_s):
        # A plausible spoken answer, so an offline voice walk-through moves the interview on.
        return CompletionResult(text=("I led the relaunch of our snacks range myself. I moved thirty percent of "
                                      "the budget to lapsed buyers and revenue grew eighteen percent in two quarters."),
                                provider="simulated", model="simulated")

    def speak(self, text, *, model, voice, timeout_s):
        return simulated_speech(text)


def simulated_speech(text: str) -> bytes:
    """A real, playable WAV (soft voice-like hum, ~0.3 s per word) so the call UI, the
    orb and sentence-by-sentence playback can be exercised offline. Never used in production."""
    import io
    import math
    import struct
    import wave
    words = max(1, len((text or "").split()))
    seconds = min(6.0, max(0.6, words * 0.3))
    rate = 16000
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            t = i / rate
            env = 0.5 + 0.5 * math.sin(2 * math.pi * 3.2 * t)  # syllable-like rhythm
            v = env * (0.6 * math.sin(2 * math.pi * 180 * t) + 0.3 * math.sin(2 * math.pi * 360 * t))
            frames += struct.pack("<h", int(v * 6000))
        w.writeframes(bytes(frames))
    return buf.getvalue()
