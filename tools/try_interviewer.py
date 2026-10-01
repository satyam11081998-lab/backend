"""
Talk to the production interviewer (services/interview_engine) in a terminal.

Same code path as the website's /messages and /voice-decision: signals -> gates ->
response function -> JSON control packet -> model. Uses your .env (OPENAI_API_KEY).
Writes NOTHING to Supabase (no attempt rows, no ai_usage rows); --case-id only reads
one case.

    python -m tools.try_interviewer                    (built-in profitability case)
    python -m tools.try_interviewer --case-id <uuid>   (a real case, read-only)
    python -m tools.try_interviewer --channel voice    (voice rules: silence, 1 sentence)

Each reply is preceded by the decision:  [function | lane | how it was worded]
Commands:  /voice  /text  /state  /reset  /quit
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    # override=True: the backend's .env wins over an old OPENAI_API_KEY left in the
    # Windows/user environment (load_dotenv never replaces an existing variable otherwise).
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
                override=True)
except Exception:  # noqa: BLE001
    pass
os.environ["ADAPTIVE_INTERVIEWER"] = "true"

import services.interview_engine as ie  # noqa: E402
from services.interviewer_decision import update_session_state  # noqa: E402
from services.session_signals import compute_signals  # noqa: E402

ie.log_ai_usage = lambda **kw: None  # no ai_usage rows from a terminal session

DEFAULT_CASE = {
    "type": "profitability",
    "content": ("A packaged-snacks maker runs at an 8% profit margin. Its costs, as a share of "
                "revenue: raw materials 50%, labour 20%, overhead 15%, logistics 15%. Raw material "
                "prices fall 10%, and an efficiency programme cuts labour and overhead by 5%. What "
                "happens to the profit margin, and what else would you recommend?"),
}


def load_case(case_id):
    if not case_id:
        return DEFAULT_CASE
    from services.supabase_client import get_supabase_client
    rows = (get_supabase_client().table("cases").select("id,type,content,market")
            .eq("id", case_id).limit(1).execute().data or [])
    if not rows:
        sys.exit(f"No case {case_id}")
    try:
        from services.markets import llm_case_content
        content = llm_case_content(rows[0])
    except Exception:  # noqa: BLE001
        content = rows[0].get("content") or ""
    return {"type": rows[0].get("type") or "", "content": content}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id")
    ap.add_argument("--channel", choices=["text", "voice"], default="text")
    args = ap.parse_args()
    if not os.getenv("OPENAI_API_KEY", "").startswith("sk-"):
        sys.exit("OPENAI_API_KEY not found - run this from the backend folder that has your .env")
    case = load_case(args.case_id)
    channel = args.channel
    transcript, state = [], {}
    print("\nCASE:\n" + case["content"] + "\n")
    print(f"Channel: {channel}. Type your turn and press Enter (one line = one turn). /quit to stop.\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text in ("/voice", "/text"):
            channel = text[1:]
            print(f"  (channel = {channel})")
            continue
        if text == "/state":
            print(f"  {state}")
            continue
        if text == "/reset":
            transcript, state = [], {}
            print("  (new session)")
            continue
        ctl = {}
        try:
            out = "".join(ie.stream_interviewer_reply(
                case["content"], case["type"], list(transcript), text,
                control_out=ctl, channel=channel, prior_state=state))
        except ie.InterviewEngineError as e:
            print(f"  [error] {e}")
            continue
        how = ("silent" if ctl.get("render") == "silent" else
               "fallback hand-back" if ctl.get("fallback") else
               "fixed short line" if ctl.get("render") == "fast" else "model, from the control packet")
        print(f"  [{ctl.get('function')} | {ctl.get('lane')} | {how}]")
        line = out.strip()
        if line.startswith("{"):
            try:
                import json
                line = json.loads(line)["data"]["text"]
            except Exception:  # noqa: BLE001
                pass
        print(f"Interviewer: {line}\n" if line else "Interviewer: (stays silent)\n")
        sig = compute_signals(transcript, text, "coached", state)
        state = update_session_state(state, ctl.get("tag") or {}, sig, function=ctl.get("function"))
        transcript.append({"role": "user", "content": text})
        if line:
            transcript.append({"role": "assistant", "content": line})


if __name__ == "__main__":
    main()
