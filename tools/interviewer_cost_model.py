"""
Cost model for the unified interviewer (and the V11 baseline) on scripted interviews.

Counts, per interview: turns by lane, model calls (generation + assessor), and ESTIMATED
tokens (prompt characters / 4 - no tokenizer is reachable offline, so every token figure
is an estimate). Prices are the ones the backend's own spend ledger uses
(services/ai_usage.py PRICES) plus the provider list prices checked on 2026-09-29.
Audio minutes are estimated from line length (~15 characters per spoken second).

Real per-interview spend must be read from ai_usage_log after real sessions; this model is
for comparing designs and for spotting anything that bills where it should not.

    python -m tools.interviewer_cost_model
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["INTERVIEWER_TELEMETRY"] = "off"
for k, v in {"OPENAI_API_KEY": "sk-test", "SUPABASE_URL": "https://x.supabase.co", "SUPABASE_SERVICE_ROLE_KEY": "t"}.items():
    os.environ.setdefault(k, v)

from tests.interviewer_fakes import FakeLLM  # noqa: E402
from services.interviewer import engine  # noqa: E402
from services.interviewer.assessor import Assessment  # noqa: E402
from services.interviewer.types import CaseContext, Channel, Lane, TurnInput  # noqa: E402

PRICE = {
    "gpt-4o-mini": (0.15, 0.60),                 # $/1M in, out
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "realtime_audio_out_per_1M": 64.00,          # gpt-realtime-2.1 audio output
    "realtime_text_in_per_1M": 4.00,             # instructions of an out-of-band response
    "realtime_audio_in_per_1M": 32.00,
    "realtime_audio_in_cached_per_1M": 0.40,
    "whisper_per_min": 0.006,
    "groq_whisper_per_min": 0.000667,
    "live_transcribe_per_min": 0.017,
    "tts1_per_min": 0.015,
}
CASE = CaseContext(case_type="guesstimate", content=("Estimate the number of new cars sold in India per year. "
                                                     "The client is a mid-size automaker planning capacity.") * 3)

SCRIPTS = {
    "strong_10": ["Hi", "Are we sizing new cars only, passenger vehicles?",
                  "I'd go top-down from households: population, households, affordability, replacement plus first-time buyers. That's my structure.",
                  "1.4 billion people, 4.5 per household, so about 31 crore households.",
                  "Top 20 percent can afford a car, about 6 crore households.",
                  "With a 7 year replacement cycle that's roughly 9 lakh replacements a year.",
                  "First-time buyers maybe 30 lakh a year.", "Plus some fleet demand, say 2 lakh.",
                  "Sanity check: 41 lakh vs about 4 million reported, fine.",
                  "My final estimate is about 41 lakh new cars a year."],
    "average_10": ["Hi", "What population should I use?", "Okay, 1.4 billion.", "let me think",
                   "Households are 1.4 billion / 4.5 = 31 crore.", "Shall I proceed?", "Maybe 20% can afford a car?",
                   "So 6 crore households.", "Replacement every 7 years, 9 lakh a year.", "My final estimate is 40 lakh."],
    "weak_10": ["Hi", "I don't know where to start.", "Can you give me a hint?", "Still stuck.", "Oh right, households.",
                "1.4 billion / 4.5 = 3.1 billion households", "wait, 31 crore.", "This is irritating.",
                "Show me the correct approach.", "Okay, 40 lakh then. That's my final answer."],
    "mixed_20": ["Hi", "What population should I use?", "Can I assume India only?", "let me think",
                 "I'd split by urban and rural, then income, then ownership. That's my structure.", "Shall I proceed?",
                 "Urban is 35%, so about 11 crore households.", "Can you give me a hint?", "Oh right, affordability.",
                 "So maybe 25% of urban households.", "50%", "Hmm", "That's 2.7 crore households.",
                 "Do we have data on replacement cycles?", "Okay, 7 years.", "So 4 lakh a year from urban.",
                 "Rural adds maybe 10 lakh.", "I'm not sure about first-time buyers.", "Say 25 lakh.",
                 "So my final estimate is 40 lakh cars a year."],
}


def est_tokens(chars: int) -> int:
    return max(1, chars // 4)


class CountingLLM(FakeLLM):
    def __init__(self):
        super().__init__("good")
        self.tok_in = 0
        self.tok_out = 0
        self.tok_out_cap = 0   # the max_tokens cap: an upper bound on real output

    def complete(self, messages, *, max_tokens):
        text, meta = super().complete(messages, max_tokens=max_tokens)
        self.tok_in += est_tokens(sum(len(m["content"]) for m in messages))
        self.tok_out += est_tokens(len(text))
        self.tok_out_cap += max_tokens
        return text, meta

    def stream(self, messages, *, max_tokens, meta):
        out = "".join(super().stream(messages, max_tokens=max_tokens, meta=meta))
        self.tok_in += est_tokens(sum(len(m["content"]) for m in messages))
        self.tok_out += est_tokens(len(out))
        self.tok_out_cap += max_tokens
        yield out


def run_brain(script, channel):
    llm = CountingLLM()
    assess_calls = {"n": 0, "tok_in": 0}

    def assess(**kw):
        assess_calls["n"] += 1
        assess_calls["tok_in"] += est_tokens(len(kw["case_content"][:1800]) + len(kw["text"]) + 700)
        return Assessment(material=False)

    st, transcript = {}, []
    lanes = {"NO_OUTPUT": 0, "PRESENCE": 0, "SUBSTANTIVE": 0}
    spoken_chars = 0
    spoken_lines = 0
    for msg in script:
        p = engine.decide_turn(turn=TurnInput(text=msg, channel=channel), case=CASE, transcript=transcript,
                               session_state={"brain": st}, assess=assess, llm=llm)
        lanes[p.decision.lane.value] += 1
        if p.lane != Lane.NO_OUTPUT:
            text = engine.word_complete(p) if channel == Channel.VOICE else "".join(engine.word_stream(p))
        else:
            text = None
        st = engine.finalize(p, text)
        transcript.append({"role": "user", "content": msg})
        if text:
            transcript.append({"role": "assistant", "content": text})
            spoken_chars += len(text)
            spoken_lines += 1
    gen_calls = len(llm.calls)
    return {"turns": len(script), "lanes": lanes, "generation_calls": gen_calls, "assessor_calls": assess_calls["n"],
            "gen_tokens_in_est": llm.tok_in, "gen_tokens_out_est": llm.tok_out, "gen_tokens_out_cap": llm.tok_out_cap,
            "assessor_tokens_in_est": assess_calls["tok_in"], "assessor_tokens_out_est": 30 * assess_calls["n"],
            "interviewer_spoken_chars": spoken_chars, "interviewer_lines": spoken_lines}


def run_v11(script, channel):
    """Baseline V11 routing on the same script (decisions only; same token estimates)."""
    from services.session_signals import compute_signals, needs_contextual_assessment
    from services.interviewer_decision import evaluate_intervention_gate
    transcript = []
    lanes = {"SILENCE": 0, "PRESENCE": 0, "SUBSTANTIVE": 0}
    assessor = gen = 0
    for msg in script:
        s = compute_signals(transcript, msg, "coached", {}, channel="text" if channel != Channel.VOICE else "voice")
        if needs_contextual_assessment(s):
            assessor += 1
        lane, mode, reason = evaluate_intervention_gate(s)
        lanes[lane] += 1
        if lane == "SUBSTANTIVE":
            gen += 1
        transcript.append({"role": "user", "content": msg})
        transcript.append({"role": "assistant", "content": "x" * 120 if lane != "SILENCE" else ""})
    return {"lanes": lanes, "generation_calls": gen, "assessor_calls": assessor}


def dollars(r, provider="gpt-4o-mini", out_key="gen_tokens_out_cap"):
    """Output priced at the max_tokens cap (upper bound): the fake model's own replies are
    shorter than real ones, so pricing them would understate spend."""
    pin, pout = PRICE[provider]
    gen = r["gen_tokens_in_est"] * pin / 1e6 + r[out_key] * pout / 1e6
    ass = r["assessor_tokens_in_est"] * PRICE["gpt-4o-mini"][0] / 1e6 + r["assessor_tokens_out_est"] * PRICE["gpt-4o-mini"][1] / 1e6
    return round(gen, 5), round(ass, 5)


def main():
    out = {}
    for name, script in SCRIPTS.items():
        for channel in (Channel.TEXT, Channel.VOICE):
            r = run_brain(script, channel)
            g_mini, a_cost = dollars(r, "gpt-4o-mini")
            g_llama, _ = dollars(r, "llama-3.3-70b-versatile")
            # Voice: every spoken line is an out-of-band response. Audio out ~1 token / 50 ms of
            # speech; the line is sent as instructions (~60 fixed tokens + the line) at the
            # text-input price. Real model lines run ~90-160 characters; use the larger of
            # the scripted length and 130 characters per model-worded line.
            model_lines = r["generation_calls"]
            est_chars = max(r["interviewer_spoken_chars"], r["interviewer_spoken_chars"] + model_lines * 100)
            spoken_s = est_chars / 15.0
            realtime_out = (spoken_s * 20 * PRICE["realtime_audio_out_per_1M"]
                            + (r["interviewer_lines"] * 60 + est_chars / 4) * PRICE["realtime_text_in_per_1M"]) / 1e6
            v11 = run_v11(script, channel)
            out[f"{name}/{channel.value}"] = {
                **r, "usd_text_model_gpt4omini": g_mini, "usd_text_model_llama70b": g_llama, "usd_assessor": a_cost,
                "usd_realtime_voice_out_est": round(realtime_out, 5) if channel == Channel.VOICE else None,
                "model_calls_per_turn": round((r["generation_calls"] + r["assessor_calls"]) / r["turns"], 2),
                "v11_baseline": {**v11, "model_calls_per_turn": round((v11["generation_calls"] + v11["assessor_calls"]) / r["turns"], 2)},
            }
    print(json.dumps({"prices": PRICE, "note": "tokens are chars/4 estimates; audio seconds ~ chars/15",
                      "interviews": out}, indent=1))


if __name__ == "__main__":
    main()
