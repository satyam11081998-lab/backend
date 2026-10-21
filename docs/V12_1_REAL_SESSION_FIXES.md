# V12.1 — fixes from the 15-day review of real sessions (candidate branch)

Source: every attempt from 2026-10-07 to 2026-10-21 (79 attempts, 38 with candidate
turns, 330 candidate turns), read turn by turn. Regression suite:
`tests/test_v121_real_sessions.py` (each fixture is a real message, typos kept).

## What was going wrong

| # | Failure class | Real examples (what the interviewer sent) |
|---|---|---|
| 1 | Questions and requests got a fixed word | "Show me the case study..." -> "Okay."; "Are we done?" -> "Let's move on to the next phase."; "i want to exist the interview i am not prepared" -> "Sure, continue."; "Budget was fixed ... or what was the budget trends" -> "Alright."; "So, this is my final answer" -> "Take it from there." |
| 2 | Wrong numbers rubber-stamped, right numbers called wrong | 1.4B / 4 = "35 million households" -> "Right."; 300M + 400M = "600M"; "35 crore is 35 million"; 6 + 2 + 1 lakh = "7 lakh"; weekly commute = daily x 7; ticket price +1.3% (should be ~8.5%) -> "solid approach"; 24.5x3 + 14.7x2 + 4.9 = 107.8 (correct) -> "there might be some confusion". |
| 3 | Data refused | "isn't explicitly stated in the case", "aren't detailed in the information we have", "I don't have specific data or an exhibit... make reasonable assumptions". |
| 4 | Phone users could not see the case | On mobile the brief sits behind the menu icon; "I can't see it on my screen", "Show me the case", "How it work". |
| 5 | Stuck learners were quizzed, not taught | "i dont know" x3, "nothing i dont know", "but i dont get it", "Clueless here" -> another open question. Scattered ideas (Infosys) -> "interesting perspective" x4, never a structure. |
| 6 | No proper ending | Final answers got "Okay." or "Right."; "did I pass?" -> "Good job!"; "how close am I to the actual value" -> "check available data". |
| 7 | Voice | Switched to Spanish after mishearing; "For more information visit www.fema.gov", "Thank you for watching" saved and scored as the candidate's words; "[Inaudible/Waiting for candidate response]" spoken/saved; "Papulo-pustular rosacea" for "population"; same question re-asked three times. |

On the 330 real candidate turns, text turns answered by the fixed-word lane went from
**51 to 2** (the 2 left are "i will go with the demand approach" -> "Go ahead." and a bare "yes").

## What changed

**Numbers are checked by code** — `services/numbers_check.py`
- Deterministic: every explicit equation ("a op b = c", "x% of y is z", brackets, mixed
  lakh/crore/M/B/k), unit conversions ("35 crore is 35 million"), listed parts vs a
  stated total, households from a population, a city/country population against a
  reference range, an office week built as x 7, a count that turns into litres, a bare
  number that finishes the candidate's own earlier expression. A slip is reported only
  when every reading of a bare number fails (precision over recall).
- Verified arithmetic is passed to the model as `arithmetic_check: verified`, so the
  interviewer may say the MATHS checks out (never that an assumption is right).
- Audit (text only, numeric derived turns): one JSON call where the model only writes
  the expression; Python evaluates it. The stated number must appear verbatim in the
  message. Model `INTERVIEWER_AUDIT_MODEL` (default gpt-4o); kill switch
  `INTERVIEWER_NUMBERS_AUDIT=off`.
- On the 330 real turns the deterministic checker flags exactly the 9 real slips and
  nothing else. On 152 adversarial correct lines it flags nothing.

**Every question gets a real answer** — `session_signals.py`, `interviewer_decision.py`
- New response functions: RESTATE_CASE, EXPLAIN_FORMAT, EXPLAIN_CONCEPT, TEACH_STEP,
  WRAP_UP_EARLY, FINAL_REVIEW, SESSION_STATUS, FOLLOW_UP_ANSWER.
- `_never_canned()`: a question or request never reaches the fixed-phrase lane or a
  question-stripping presence beat; in chat, a short answer to the interviewer's own
  question gets FOLLOW_UP_ANSWER, and a turn with content gets a model-worded line.
- The context assessor can no longer turn a real question into "answering", and is
  skipped when a deterministic route already decided the turn.
- "help" counts only as a request ("it would be really helpful" no longer triggers a hint).

**Teaching** — TEACH_STEP names the approach ("top-down: population, then the share who
buy, then how often"), lays the skeleton out on short lines joined by arrows (chat
only: `visual_structure_allowed`), does the first step with the candidate and hands
over the next one. Triggers: "i dont know"/"clueless", "teach me / walk me through",
two help asks in a row, explicit frustration, help with nothing on the table, and 3+
free-floating ideas with no structure (groups THEIR ideas into buckets). An explicit
"I want a hint" still gets a hint.

**Endings** — FINAL_REVIEW: their number back, one sanity check computed in code
(`implied_check`, e.g. "one for every 250 people"), one strength, one thing to
sharpen, and "tap Submit". SESSION_STATUS for "are we done / did I pass / how close am
I"; once a final answer exists, the stored model solution is offered as
`reference_answer` (never before). WRAP_UP_EARLY lets an unprepared candidate leave
kindly and offers "teach me" instead.

**What ships** — a data refusal is regenerated once and, if it survives, the refusing
sentence is dropped; a no-question reply that was only a question (which used to ship
as "Let's move on to the next phase.") or only praise is regenerated once; the
streamed lane now drops a praise opener and cuts a stacked second question;
"Great start, but there's..." becomes "There's..."; line breaks survive sanitising.

**Phone users** — the opening note now says "On a phone, tap the menu at the top left to
read the full brief"; the interviewer is told where the brief is and gives it back
when asked.

**Voice**
- `routes/realtime.py`: default voice `marin` (realtime-native, natural; `REALTIME_VOICE`
  still overrides), `noise_reduction: near_field` (`REALTIME_NOISE_REDUCTION`), a
  case-vocabulary transcription prompt (`REALTIME_TRANSCRIBE_PROMPT`). If the API rejects
  the extras, the session retries without them; a rejected transcription model still
  falls back to whisper-1.
- Playbook: English only always; ASR words that make no sense are read as the nearest
  case word; transcription noise ignored; no stage directions; answer their question and
  stop (no re-asking); phone case location + spoken recap; name the approach when lost;
  silently check every calculation; final answer = one plain-number sanity check + submit.
- `services/transcript_hygiene.py`: ASR hallucinations (URLs at the end, "thank you for
  watching", subtitle credits) and bracketed stage directions are removed before a voice
  turn is saved, answered or scored; an empty result is metered but not saved.

## Adversarial review (before commit)
An independent review tried to break the build. What it found and what changed:
- The checker called correct work wrong in 25 of 50 synthetic lines (conversions with nouns,
  rates or currencies; fiscal years like "2023-24"; "24/7"; ranges like "10-15%"; "5% of 20%
  of"; "2 cr + 10% growth"; ml->litres and minutes->hours inside a chain; a money "total"
  after counted parts). All of these shapes now stay quiet. 0 of the 152 correct lines are
  flagged now (pinned in `tests/fixtures/v121_correct_working.json`), and the 9 real slips
  are still caught.
- Ordinary case talk was misrouted: "the client wants to quit the market", "customers give up
  on the brand", "stop the practice of 60-day credit", "To summarize the case: ...", "My
  estimate is that 30% ...", "any feedback from customers?". Exit, final, status, restate and
  teach signals are now tied to the speaker or the session, and a final answer needs a figure.
- The stored model answer could leak after a misread final. It is now only the headline figure
  ("about 5 lakh laptops"), only after a real final answer, and never in exam policy.
- Pathological 20,000-character inputs could take seconds. Parsing is now linear and capped,
  and every check is guarded so it can never raise inside a turn.
- Sanitising no longer rewrites "So far, ..." or "Though small, ...". The stream guard keeps line
  breaks whatever the chunk boundaries. A refusal that also gives a figure is kept.
- Hinglish and noun-phrase questions ("market share kitna hai", "And the margins", "I would
  like to know ...") are now answered.
- The voice playbook no longer asks the realtime model to recompute every number. It points
  only at clearly wrong steps. The transcription vocabulary follows the market (US vs India).

**Latency note.** The numbers audit is one gpt-4o call (4 s timeout, no retries) that runs
before the reply. It fires only on chat turns that state a NEW derived figure that Python
could not check itself. That is about a quarter of such turns in the synthetic sets, and fewer
in real sessions. To turn it off: `INTERVIEWER_NUMBERS_AUDIT=off`.

## Arena adversarial run (Oct 21) — the 16 scenarios New still failed
Run: 50 scenarios, chat, gpt-4o judge. Current (main) 30, New (candidate @ 653f2a0) 34,
5 regressions, 9 improvements. Every scenario New still failed is now pinned in
`tests/test_v121_arena_adversarial.py`.

| Scenario | What New did | What changed |
|---|---|---|
| help_when_stuck_with_work (R) | generic profit tree, too long | help with real working on the table gets a hint ON that working; Python solves their equation (`0.95S - 2 = 0.80(S-2)` -> S ≈ 2.67, verified) so the interviewer can say it holds. Teaching from scratch only after a hint on it failed. |
| wants_solution_outright (R), semantic_repetition_guard | right content, flagged too long | numbered lists become arrow lines; prose capped at 5 sentences (non-stream) and 4 + the final question (stream, `StreamShapeGuard`); sentence targets lowered. |
| correct_but_uncertain_do_not_derail (R) | "let's refine it, break it down" | FOLLOW_UP_ANSWER: an unsure but workable figure is carried forward, never interrogated. |
| asks_same_thing_twice (R) | repeated "start with the population" | `already_tried` = the pointer that did not land; TEACH_STEP switches method (bottom-up from one household / shop / day). |
| sanity_check_absent (R) | invented "0.5 orders per person, plausible" | "small town" / "village" size ranges; `implied_verdict` ("looks high") for orders a day per person, durables per person, more users than people; an implausible or unchecked guesstimate final gets ONE sanity check before the close. |
| factor_1000_unit_error | "corrected" to 0.5 crore litres (also wrong) | unit restatements checked in code: "5 crore ml is 50,000 litres (1,000 ml make a litre)". |
| double_count_error | vague | `services/logic_check.py`: the whole population added on top of its own segments. |
| confidently_wrong_penetration | "Does that number look right?" | "100% / everyone buys it daily" named; SANITY_CHECK must test it against a real counter-case. |
| contradicts_established_fact | "That's an interesting insight" | established facts (brief, interviewer lines, confirmed candidate lines) vs the turn's claim: "volumes are flat" vs "selling fewer units". |
| (all corrections) | statement only | CORRECT_AND_CONTINUE may ask ONE targeted question ("How many ml make a litre?"). |
| explain_why_meta | repair (assessor said stuck) | EXPLAIN_WHY: one line on why it matters, hand back. Decided before the assessor. |
| product_ux_question | a hint (assessor said help) | PRODUCT_HELP from `PRODUCT_FACTS`. |
| too_many_questions_back_off | more guidance | HOLD_SPACE: a fixed "Fair enough. It's all yours; I'll hold my questions."; the next turn asks nothing. |
| productive_thinking_pause | "Alright, continue." | "Take your time." |
| own_facts_competitors | "about 15 competitors" | ANSWER_DIRECT gives a usable shape (main players and their shares). |
| good_answer_should_advance | judged as praise | REFLECT_PROGRESS: neutral line + the concrete next step inside their approach. |

The voice playbook got the same four behaviours (take your time, back off, why-asked, app questions).
On the 330 real turns none of the new checks fire and no route changes; the 152 correct lines stay unflagged.

**Independent adversarial review of these fixes (before commit).** About 1,100 inputs. What it
broke in the first version, now fixed and pinned in the test file:
- Correct algebra called wrong: scales and comma groups inside an equation ("4N = 1.2 crore,
  N = 30 lakh", "5P = 25,000"), unit letters read as unknowns ("1 M = 10 L, so 5 M = 50 L",
  "1 h = 60 min"). Equations are now solved in real units, unit letters are never unknowns,
  "so 5 M = 50" is never taken as the solution, and a solved equation overrides the chain
  reader's fragments inside it ("(1-0.2) = 0.8, not 80").
- The contradiction check fired on correct reasoning (23 of 62): negations after the claim,
  self-corrections, regions and channels, competitors, other periods, goals, comparisons,
  other cost lines. It now needs a claim about OUR total in this period, skips all of those,
  ignores industry facts, and never contradicts on costs (too many sub-lines).
- Double count and "everyone daily" fired on sanity checks and business assumptions: the
  whole place must be what is added, and "everyone" must mean people in general, not shops,
  riders, a target segment, someone else's claim, or a share ("out of 100%, 40% buy").
- Plausible finals told "looks high" (rides, tickets, UPI transactions a day; passengers or
  visitors a year; phones in use): only orders/deliveries a day, durables per person and a
  count of people above the population can be called high; a final with no base closes normally.
- Case questions sent to PRODUCT_HELP / EXPLAIN_WHY ("results of the survey", "why is it
  important for the client to grow"): the app question must be about MY results / Submit /
  ending the session; a topic "why X?" must name a topic from your last line, never after "Go on.".
- Shaping cut at "Rs." and dropped the figure, broke a long sentence mid-word, lost a question
  followed by a list. The splitter now protects abbreviations, domains, ellipses and list
  markers; a sentence goes out or is held whole; the cap keeps the question; "Rs. 300" ships as
  "Rs 300". 400 generated replies x 25 chunkings: no difference by chunking, nothing over five
  sentences, no question lost.
- Exam policy no longer carries the corrected figure (in a correction or a hint on their working).
- After a space request every function is told the same thing (no question), and the first
  "i dont know" is never treated as an already-tried pointer.

## Intentional test changes
- `test_response_functions`: help with nothing on the table -> TEACH_STEP (help after work
  still MICRO_HINT); a bare number in chat is model-worded (voice keeps SHORT_ACK); the
  numbers audit is switched off in this suite (it counts the interviewer's own calls).
- `test_arena_parity` / `test_arena_isolation`: production now trims stacked questions
  and praise in streamed replies and drops refusals, so the arena shows the trimmed text
  (`enforced` is True); the fast-lane fixture is a mid-thought turn.
- `test_voice_model_led`: transcription carries the vocabulary prompt; noise reduction
  and the marin voice are pinned.
- `test_interviewer_mode` (legacy V10 vocabulary): new routes appear for help/stuck
  (TEACH), finals (FINAL_REVIEW) and "how did I do" (SESSION_STATUS). This is intended.
  After the arena run: "why are you asking" -> EXPLAIN_WHY and "where are my results" ->
  PRODUCT_HELP (the legacy file expects ANSWER_DIRECT); "too many questions" now matches
  its HOLD_SPACE. 29 legacy mismatches, all from intended routes.
- `test_arena_isolation::test_violations_are_reported_for_an_offending_reply`: the shipped
  reply is now capped at five sentences, so `too_long` is reported on the raw model text,
  not on what ships.
- Offline eval (`tools/eval_interviewer_behavior.py`): 17 signal mismatches against 16 before.
  The new one is `frustration_change_strategy`: "you're going round and round, not helping" is
  no longer read as a help request (the word "helping"). It routes to REPAIR, which is the
  behaviour the scenario asks for.

## Env switches
`INTERVIEWER_NUMBERS_AUDIT=off`, `INTERVIEWER_AUDIT_MODEL`, `REALTIME_VOICE`,
`REALTIME_NOISE_REDUCTION=off|near_field|far_field`, `REALTIME_TRANSCRIBE_PROMPT`,
`REALTIME_MODEL_LED_EAGERNESS` (try `medium` if candidates feel cut off mid-thought).
