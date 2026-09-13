# Interviewer eval harness

Stop analysing voice transcripts one at a time. This runs simulated **candidate
personas** against the **real interviewer prompt** and has an **LLM judge** score
every session against explicit must-never / must-always rules.

## Run it

```bash
export OPENAI_API_KEY=sk-...
cd consilio-backend
python eval/interviewer_eval.py                       # all personas, 6 turns each
python eval/interviewer_eval.py --turns 8
python eval/interviewer_eval.py --personas wants_ai_to_calc,give_me_answer
```

Optional model overrides (defaults are fine):

```bash
EVAL_INTERVIEWER_MODEL=gpt-4o EVAL_CANDIDATE_MODEL=gpt-4o-mini EVAL_JUDGE_MODEL=gpt-4o \
  python eval/interviewer_eval.py
```

## What you get

A console scorecard plus `eval/out/scorecard.json`, and a `eval/out/fail_<persona>.txt`
transcript for every persona that failed a check. Example:

```
no_praise            5/6 pass   leaked in: rambling_pauser
no_did_math          6/6 pass
no_echoed_numbers    4/6 pass   leaked in: wants_ai_to_calc, rambling_pauser
landed_aha           3/4 pass   missed in: give_me_answer
```

Exit code is non-zero if any check fails, so it can gate a deploy in CI later.

## The personas (each stresses a real failure we saw)

| persona            | type        | stresses |
|--------------------|-------------|----------|
| structured_strong  | guesstimate | no praise; land the aha at close |
| rambling_pauser    | guesstimate | don't echo their numbers; stay short; patience |
| wants_ai_to_calc   | guesstimate | don't do the maths; don't echo numbers |
| give_me_answer     | case        | give a foothold when stuck; don't cave and solve it |
| asks_unspecified   | case        | never say "isn't specified"; own the facts |
| adversarial        | case        | stay in character vs jailbreak |

Add or edit personas in the `PERSONAS` dict at the top of `interviewer_eval.py`.
Add or edit checks in `MUST_NEVER` / `MUST_ALWAYS`.

## What it does NOT test

Audio turn-taking / interruption (the "it doesn't wait, it repeats" problem) is a
different layer — handled by the realtime turn-detection engine (`semantic_vad` in
`routes/realtime.py`) and by monitoring a sample of real calls. This harness tests
the interviewer's **textual behaviour** (the prompt) at scale, cheaply. Run it after
any change to `prompts/interview_prompts.py` or the interviewer model.
