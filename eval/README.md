# Interviewer eval harness

Stop analysing voice transcripts one at a time. This runs simulated **candidate
personas** against the **real interviewer prompt** and has an **LLM judge** score
every session against explicit must-never / must-always rules.

## Watch it live (5 sessions)

See the interviewer actually work — every turn printed as it happens, then the
judge's verdict. Your key is read from the backend `.env` automatically, so this
is the whole setup (one line):

```
cd consilio-backend
python eval/interviewer_eval.py --show --runs 1 --personas structured_strong,rambling_pauser,wants_ai_to_calc,give_me_answer,asks_unspecified
```

That's 5 sessions — 3 guesstimates (a strong candidate, a rambler, one who wants
the AI to do the maths) and 2 cases (one who begs for the answer, one who asks for
unspecified facts). `--show` prints each candidate/interviewer exchange live and
runs one at a time so it reads like a real transcript. Takes a couple of minutes.
(If a module is missing: `pip install httpx python-dotenv`.)

## Run the full scorecard

```bash
export OPENAI_API_KEY=sk-...
cd consilio-backend
python eval/interviewer_eval.py                       # 6 personas x 10 runs = 60 sessions
python eval/interviewer_eval.py --runs 3              # quicker: 6 x 3 = 18 sessions
python eval/interviewer_eval.py --concurrency 6       # run more sessions in parallel
python eval/interviewer_eval.py --personas wants_ai_to_calc,give_me_answer --runs 10
```

Each persona runs `--runs` times (default **10**) so one lucky/unlucky sample
doesn't decide the verdict — you get "praise leaked in 3/10", not a coin flip.
60 sessions makes a lot of API calls and takes several minutes; `--concurrency`
(default 4) runs several at once, and `--runs 3` is a fast smoke check.

Optional model overrides (defaults are fine):

```bash
EVAL_INTERVIEWER_MODEL=gpt-4o EVAL_CANDIDATE_MODEL=gpt-4o-mini EVAL_JUDGE_MODEL=gpt-4o \
  python eval/interviewer_eval.py
```

## What you get

A console scorecard plus `eval/out/scorecard.json`, and a `eval/out/fail_<persona>.txt`
transcript for every persona that failed a check. Example:

```
check                 pass  fail   na   leaked in (fails/runs)
no_rubber_stamp         55     5    0   structured_strong(5/10)
no_repetition           49    11    0   give_me_answer(6/10), rambling_pauser(5/10)
no_echoed_numbers       48    12    0   wants_ai_to_calc(7/10), rambling_pauser(5/10)
no_hints_or_solutions   51     9    0   give_me_answer(6/10), rambling_pauser(3/10)
```

Exit code is non-zero if any check fails, so it can gate a deploy in CI later.

## The personas (each stresses a real failure we saw)

| persona            | type        | stresses |
|--------------------|-------------|----------|
| structured_strong  | guesstimate | no rubber-stamp (light affirmation ok); no teaching at close |
| rambling_pauser    | guesstimate | don't echo numbers; don't loop/repeat; stay short |
| wants_ai_to_calc   | guesstimate | don't do the maths; don't echo numbers |
| give_me_answer     | case        | no hints even when stuck/begging; don't cave and solve it |
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
