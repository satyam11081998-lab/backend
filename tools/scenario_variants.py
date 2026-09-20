"""Paraphrase bank for the interviewer eval.

Each eval scenario gets several candidate-message paraphrases in the same messy,
code-mixed student voice, preserving the SITUATION (so it should route to the same
interviewer MODE). The live eval samples one at random each run, so the score is not
a memorised pass over 50 fixed strings -- it tests whether the deterministic routing
GENERALISES across phrasings. `python -m tests.test_interviewer_mode` runs every
variant offline and flags any that routes to an unacceptable mode.

ACCEPTABLE[id] is the set of interviewer moves that are behaviourally fine for that
situation (usually one; a few situations have two equally-valid moves).
"""
from __future__ import annotations

# scenario id -> extra candidate-message phrasings (the original `new` is used too)
VARIANTS = {
    "help_when_stuck_with_work": [
        "yaar i really can't crack this, help me na",
        "i'm totally blank on this, not getting it at all",
        "bro help, i have no idea what's going wrong here",
    ],
    "reasonable_assumption_stands": [
        "let's say around 2 cups per person a day, and maybe half of people drink chai",
        "i'll assume 2 cups daily each, and roughly 50% are chai drinkers",
        "ok so about 2 cups a day per head, and 50% chai drinkers roughly",
    ],
    "material_unit_error": [
        "46 cr bottles a month times 12 is 552 cr litres a year, that's my final answer",
        "so my final number is 552 crore litres a year from the bottles",
        "final answer, 552 cr litres annually, 46 cr bottles into 12 months",
    ],
    "wants_to_stop_frustrated": [
        "ugh this is annoying, i don't want to continue",
        "so irritating yaar, leave it na",
        "nah i'm done with this, let's just stop",
    ],
    "wants_solution_outright": [
        "yaar i can't do this, just give me the approach na",
        "not able to solve, provide the approach please",
        "show me the approach please, i'm lost",
    ],
    "repair_after_interviewer_repeat": [
        "wait what are we actually solving here",
        "i'm confused, what even is the question",
        "honestly what is the question we're doing",
    ],
    "meta_identity_probe": [
        "hold on, are you an ai or a real person?",
        "wait are you chatgpt or something?",
        "are you a bot or an actual interviewer?",
    ],
    "scope_question_own_facts": [
        "roughly what's the city's population?",
        "what's the population here approximately?",
        "how many people live in this city roughly?",
    ],
    "voice_noise_asr": [
        "haan matlab ummmmm",
        "toh phir uhhhhhh",
        "hmmmmm woh ya",
    ],
    "minor_error_do_not_nitpick": [
        "so about 10 crore households, i'll just round to that",
        "roughly 10 cr households, rounding it off",
        "let's call it 10 crore households, approx",
    ],
    "good_answer_should_advance": [
        "i'd break cost into fixed and variable, check which moved, probably variable like raw material",
        "let me split costs fixed vs variable and see the mover, guessing raw material spike",
        "cost side fixed and variable, i think variable moved, the raw material side",
    ],
    "frustration_change_strategy": [
        "see you're just going in circles, this isn't helping",
        "you keep going round and round, not helping at all",
        "this is going nowhere, you're not helping me here",
    ],
    "stuck_early_no_work": [
        "honestly i have no idea how to even begin this",
        "i don't know where to start here at all",
        "no idea how to start this whole thing",
    ],
    "double_count_error": [
        "and then i'll add the whole city's population eating out on top, sum it all",
        "plus add everyone in the city eating out, add them all up",
        "then also add the whole population eating out and total everything",
    ],
    "rubber_stamp_weak_reasoning": [
        "i guess it's just the market being bad and competition, that's all it is honestly",
        "probably just the bad market and tough competition, nothing else to it really",
        "it's mostly the weak market and competition pressure i think, that's it",
    ],
    "factor_1000_unit_error": [
        "so total is 5 crore ml which is basically 5 crore litres for the city, done",
        "5 crore ml, that's 5 crore litres for the whole city, final",
        "the total comes to 5 crore ml so 5 crore litres overall, done",
    ],
    "tiny_arithmetic_slip_ignore": [
        "so 12 into 4 is about 46, call it 46 lakh roughly",
        "12 times 4 is roughly 46, i'll say 46 lakh approx",
        "12 into 4 comes around 46, about 46 lakh",
    ],
    "confidently_wrong_penetration": [
        "obviously it's 100%, literally everyone in india buys this daily, so 140 crore units a day",
        "it's definitely 100%, everyone buys this every single day, 140 crore units daily",
        "clearly 100% penetration, all of india buys this daily, demand is 140 crore a day",
    ],
    "correct_but_uncertain_do_not_derail": [
        "i think it's around 20 lakh, not fully sure but feels about right",
        "maybe 20 lakh? not totally confident but seems okay",
        "somewhere near 20 lakh i'd guess, not sure but roughly right",
    ],
    "contradicts_established_fact": [
        "so the revenue drop must be because we're selling fewer units",
        "the revenue must have dropped since we're selling less volume",
        "revenue fell so it has to be that unit sales went down",
    ],
    "broken_mece_buckets": [
        "i'll split them into students, young people, and people who buy a lot",
        "let me segment into students, youngsters, and heavy buyers",
        "buckets would be students, young folks, and people who buy a lot",
    ],
    "wrong_denominator": [
        "market share is our revenue divided by our own sales, that gives the share",
        "so market share is just our revenue over our own sales",
        "i'll take market share as revenue divided by our own sales",
    ],
    "over_help_risk_still_working": [
        "okay so first the population, then i need to think about how many are actually relevant here",
        "let me first get the population, then i need to think about the relevant chunk",
        "starting with population, then i need to figure out how many actually count",
    ],
    "repeated_stuckness_escalate": [
        "still not sure how to begin",
        "i'm still stuck on how to start",
        "yeah still no idea how to begin this",
    ],
    "recovers_after_hint_step_back": [
        "oh right, so demand is total orders and capacity is what one outlet serves, so i just divide them!",
        "ohh i see, demand is total orders, capacity is per outlet, so i divide them",
        "got it, so i just divide total demand by one outlet's capacity!",
    ],
    "candidate_re_asks_answered_fact": [
        "wait, what's the population again?",
        "sorry, remind me the population?",
        "what was the population number again?",
    ],
    "semantic_repetition_guard": [
        "no idea honestly",
        "honestly no idea at all",
        "yeah i have no idea",
    ],
    "asks_same_thing_twice": [
        "i still dont get how to start",
        "i really still don't get how to begin",
        "still not getting how to actually start",
    ],
    "explicit_hint_request": [
        "can you give me a small hint?",
        "just a tiny hint please?",
        "could you give me a little hint to start?",
    ],
    "solution_request_late": [
        "okay just tell me the final answer now, i've done most of it",
        "i've done most of it, just give me the final answer",
        "just tell me the answer now, i'm almost through it",
    ],
    "explain_why_meta": [
        "why are you asking about the value chain?",
        "why does the value chain matter here?",
        "what's the point of the value chain question?",
    ],
    "product_ux_question": [
        "wait where do i see my results after this?",
        "where will i see my results once we're done?",
        "after this, where do i find my results?",
    ],
    "own_facts_growth_rate": [
        "what's the market growth rate?",
        "how fast is the market growing?",
        "what's the growth rate of this market?",
    ],
    "own_facts_competitors": [
        "how many competitors are there in this market?",
        "how many players are in this market?",
        "how many competitors do we have here?",
    ],
    "valid_alt_customer_journey": [
        "i'd start with the customer journey, awareness then purchase then repeat, instead of a profit tree",
        "rather than a profit tree i'll use the customer journey, awareness to purchase to repeat",
        "let me use the customer journey instead of a profit tree, awareness then buy then repeat",
    ],
    "valid_alt_supply_first": [
        "i know you'd expect demand first but i want to check the supply side, our production capacity, first",
        "instead of demand first, let me start on the supply side, production capacity",
        "i'll look at supply first, our production capacity, before demand",
    ],
    "valid_alt_top_down_sizing": [
        "for the market size i'll go top-down from gdp rather than bottom-up per person",
        "i'll size it top-down off gdp instead of bottom-up per head",
        "rather than bottom-up i'll do top-down from gdp for the market size",
    ],
    "productive_thinking_pause": [
        "let me think for a second",
        "give me a moment to think",
        "hold on, let me think about this",
    ],
    "thinking_out_loud_self_correct": [
        "umm 20 million... no wait, maybe 15 million... let me redo the penetration",
        "20 million... actually no, 15 million... let me redo the penetration part",
        "hmm 20 mil, wait no maybe 15 mil, let me recompute the penetration",
    ],
    "keyboard_mash_noise": [
        "bhhhhhkkk",
        "asdkjhaskjd",
        "kkkkkkjjjjj",
    ],
    "done_take_final_answer": [
        "okay i'm done, my recommendation is to cut the delivery fee and add a night-shift staff",
        "i'm done, i'd recommend cutting the delivery fee and adding night-shift staff",
        "that's it from me, my recommendation is drop the delivery fee and add night staff",
    ],
    "small_talk_after_close": [
        "so how did i do overall, any tips",
        "how did i do, any feedback for me?",
        "any tips on how i did overall?",
    ],
    "too_many_questions_back_off": [
        "you're asking too many questions man, just let me solve",
        "too many questions yaar, let me just solve it",
        "stop asking so much, let me solve it na",
    ],
    "wants_to_move_on": [
        "eh, can we just move on to the next part",
        "let's just move on to the next part na",
        "can we move on to the next section",
    ],
    "help_after_good_work": [
        "i've got the demand but i'm stuck on what to do next",
        "i have the demand number but stuck on the next step",
        "got the demand, now i'm stuck on where to go next",
    ],
    "challenge_confident_wrong_80pct": [
        "penetration is definitely 80%, almost everyone uses premium cars",
        "it's clearly 80% penetration, nearly everyone drives premium cars",
        "penetration has to be 80%, basically everyone uses premium cars",
    ],
    "accept_well_defended_assumption": [
        "i took 30% urban because tier-1 metros skew higher and that's roughly india's urban share overall",
        "30% urban since tier-1 metros skew higher and that's about india's urban share",
        "i used 30% urban because that's india's rough urban share and metros skew up",
    ],
    "sanity_check_absent": [
        "so my final answer is 90000 orders per day for this small town",
        "final answer, 90000 orders a day for this small town",
        "my final number is 90000 orders per day in this small town",
    ],
    "greeting_kickoff": [
        "let's start",
        "okay let's begin",
        "alright, start",
    ],
    "strong_synthesis_close": [
        "so overall i'd recommend hiring one more night-shift person and a small delivery fee, that fixes throughput without hurting demand",
        "overall my recommendation is one more night-shift hire plus a small delivery fee, fixes throughput without hurting demand",
        "to sum up, i'd add a night-shift person and a small delivery fee, throughput fixed without denting demand",
    ],
}

# behaviourally-acceptable interviewer move(s) per situation (a few allow two)
ACCEPTABLE = {
    "help_when_stuck_with_work": {"HINT"},
    "reasonable_assumption_stands": {"ACK_ADVANCE"},
    "material_unit_error": {"SANITY_CHECK", "CORRECT_MATERIAL"},
    "wants_to_stop_frustrated": {"CLOSE"},
    "wants_solution_outright": {"DELIVER_SOLUTION"},
    "repair_after_interviewer_repeat": {"REPAIR"},
    "meta_identity_probe": {"DEFLECT_META"},
    "scope_question_own_facts": {"ANSWER_DIRECT"},
    "voice_noise_asr": {"NOISE"},
    "minor_error_do_not_nitpick": {"ACK_ADVANCE"},
    "good_answer_should_advance": {"ACK_ADVANCE", "PROBE"},
    "frustration_change_strategy": {"REPAIR"},
    "stuck_early_no_work": {"HINT"},
    "double_count_error": {"CORRECT_MATERIAL"},
    "rubber_stamp_weak_reasoning": {"ACK_ADVANCE", "PROBE"},
    "factor_1000_unit_error": {"CORRECT_MATERIAL"},
    "tiny_arithmetic_slip_ignore": {"ACK_ADVANCE"},
    "confidently_wrong_penetration": {"CHALLENGE_CLAIM"},
    "correct_but_uncertain_do_not_derail": {"ACK_ADVANCE"},
    "contradicts_established_fact": {"ACK_ADVANCE", "PROBE", "CORRECT_MATERIAL"},
    "broken_mece_buckets": {"CORRECT_MATERIAL"},
    "wrong_denominator": {"CORRECT_MATERIAL"},
    "over_help_risk_still_working": {"HOLD_SPACE", "ACK_ADVANCE"},
    "repeated_stuckness_escalate": {"HINT"},
    "recovers_after_hint_step_back": {"RELEASE"},
    "candidate_re_asks_answered_fact": {"ANSWER_DIRECT"},
    "semantic_repetition_guard": {"REPAIR", "HINT"},
    "asks_same_thing_twice": {"HINT", "REPAIR"},
    "explicit_hint_request": {"HINT"},
    "solution_request_late": {"DELIVER_SOLUTION"},
    "explain_why_meta": {"ANSWER_DIRECT"},
    "product_ux_question": {"ANSWER_DIRECT"},
    "own_facts_growth_rate": {"ANSWER_DIRECT"},
    "own_facts_competitors": {"ANSWER_DIRECT"},
    "valid_alt_customer_journey": {"ACK_ADVANCE"},
    "valid_alt_supply_first": {"ACK_ADVANCE"},
    "valid_alt_top_down_sizing": {"ACK_ADVANCE"},
    "productive_thinking_pause": {"HOLD_SPACE"},
    "thinking_out_loud_self_correct": {"HOLD_SPACE"},
    "keyboard_mash_noise": {"NOISE"},
    "done_take_final_answer": {"CLOSE"},
    "small_talk_after_close": {"CLOSE"},
    "too_many_questions_back_off": {"HOLD_SPACE"},
    "wants_to_move_on": {"ACK_ADVANCE"},
    "help_after_good_work": {"HINT"},
    "challenge_confident_wrong_80pct": {"CHALLENGE_CLAIM"},
    "accept_well_defended_assumption": {"ACK_ADVANCE"},
    "sanity_check_absent": {"SANITY_CHECK"},
    "greeting_kickoff": {"OPEN"},
    "strong_synthesis_close": {"CLOSE"},
}
