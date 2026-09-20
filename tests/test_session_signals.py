"""Unit tests for the deterministic adaptive-interviewer layer (Phase 1).

Runs on the standard library alone (no OpenAI, no network):
    python -m tests.test_session_signals
Cases are seeded from REAL failed sessions in the production logs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The deterministic layer now reaches services/ai_providers.py, which imports
# the OpenAI SDK at module scope. Stub what is missing BEFORE those imports so
# this suite keeps its stdlib-only promise; no-op when the SDK is installed.
# See tests/_sdk_stubs.py for why the fix is here and not in the service.
from tests._sdk_stubs import install as _install_sdk_stubs  # noqa: E402
_install_sdk_stubs()


from services.session_signals import (  # noqa: E402
    detect_intent, compute_signals, build_signal_block, _looks_garbage,
)
from services.interviewer_decision import (  # noqa: E402
    parse_control_tag, StreamTagStripper, update_session_state, detect_violations,
)

_fail = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        _fail.append(name)


def run_stream(stripper, tokens):
    out = []
    for t in tokens:
        out.extend(list(stripper.feed(t)))
    out.extend(list(stripper.flush()))
    return "".join(out)


# ---- intent detection (from real transcripts) -------------------------------
check("help intent (Store Profit)",       detect_intent("help, I am not getting it")["intent"] == "asking_for_help")
check("solution intent (Organic Fert.)",  detect_intent("Provide me with the approach to solve this case")["intent"] == "wants_solution")
check("solution intent (show approach)",  detect_intent("Unable to get. Show me the correct approach needed")["intent"] == "wants_solution")
check("stop intent (leave it)",           detect_intent("boss ... Pls guide or leave")["frustration"] is True)
check("stop intent (don't want)",         detect_intent("You know, I don't want to")["skip_or_stop"] is True)
check("frustration (irritating)",         detect_intent("No, no, it is getting irritating.")["frustration"] is True)
check("frustration (beating bush)",       detect_intent("I think you are beating around the bush")["frustration"] is True)
check("frustration (too deep, Chai)",     detect_intent("I think you are going too into deep")["frustration"] is True)
check("meta (Chargipity/Claude/Gemina)",  detect_intent("Wait, what are you? Chargipity, Claude or Gemina?")["intent"] == "meta")
check("scope question (population)",       detect_intent("what is the population of jaipur?")["intent"] == "scope_question")
check("answering (defensible assumption)",detect_intent("2 cups of tea per day and I'll assume around 50%")["intent"] == "answering")
check("greeting only (hi)",               detect_intent("hi")["is_greeting_only"] is True)

# ---- garbage / ASR-noise detection (real) -----------------------------------
check("garbage Bbbhhhh",  _looks_garbage("Bbbhhhhhhhhhhhhhhhhh") is True)
check("garbage kloool",   _looks_garbage("kklooooooollllllllllllllllll") is True)
check("garbage yibuyb",   _looks_garbage("yibuybuybybybyibiubiub") is True)
check("garbage mash",     _looks_garbage("sdvabfgnrngfrnvd rsnzht") is True)
check("real text passes", _looks_garbage("2 cups of tea per day") is False)
check("real answer passes", _looks_garbage("population is 2.3 cr and consumption") is False)

# ---- signals: stuck learner WITH work down (Store Profit shape) --------------
sig = compute_signals(
    transcript=[
        {"role": "user", "content": "0.95S - 2 = 0.80(S - 2), solving gives 0.67 crore"},
        {"role": "assistant", "content": "How did you arrive at 2.67 crore? What steps did you take?"},
        {"role": "user", "content": "help, I am not getting it"},
        {"role": "assistant", "content": "That's what I'm here to observe. What's your next move to solve for S?"},
    ],
    new_user_message="boss I have solved it exactly the way I want. Pls guide or leave",
    teaching_policy="coached",
)
check("repair_due after failed probes + frustration", sig["repair_due"] is True)
check("frustration high", sig["frustration"] == "high")
check("has_work true", sig["has_work"] is True)
check("signal block mentions repair", "repair is due" in build_signal_block(sig))
check("signal block bans 'understand your frustration'", "never say 'I understand your frustration'" in build_signal_block(sig))

# ---- signals: reasonable assumption should NOT look stuck/frustrated ---------
sig2 = compute_signals(
    transcript=[{"role": "assistant", "content": "What's your first cut at the structure?"}],
    new_user_message="I'll take Delhi population as 2.3 crore and split by frequent/occasional/rare",
    teaching_policy="coached",
)
check("calm answering: no frustration", sig2["frustration"] == "none")
check("calm answering: repair not due", sig2["repair_due"] is False)

# ---- signals: interviewer repeated itself (Pressure Cookers) -----------------
sig3 = compute_signals(
    transcript=[
        {"role": "assistant", "content": "That's the exercise - what's your next step?"},
        {"role": "user", "content": "WHATS THE NEXT STEP?"},
        {"role": "assistant", "content": "That's the exercise - what's your next step?"},
    ],
    new_user_message="I MEAN WHAT SHOULD I FIND WHATS THE QUESTION",
    teaching_policy="coached",
)
check("interviewer_repeating detected", sig3["interviewer_repeating"] is True)

# ---- control tag parsing + stream stripping ---------------------------------
tag, cleaned = parse_control_tag("<<mode=coach; materiality=material; intervention=micro_hint; hint=1>>\n\nYou're one step away — expand the right side.")
check("tag parsed mode", tag.get("mode") == "coach")
check("tag parsed hint", tag.get("hint") == "1")
check("tag stripped from reply", cleaned == "You're one step away — expand the right side.")

check("stream strips tag", run_stream(StreamTagStripper(), ["<<mode=", "coach>>", "\n\nHello ", "there"]) == "Hello there")
check("stream no-tag passthrough", run_stream(StreamTagStripper(), ["Hello ", "there, ", "go on"]) == "Hello there, go on")
check("stream tag as single chunk", run_stream(StreamTagStripper(), ["<<mode=coach>>\n\nGo on."]) == "Go on.")
check("stream unterminated tag flushes reply", "Go on" in run_stream(StreamTagStripper(budget=20), ["<<mode=coach no close ", "and more text Go on now here"]))

# ---- Phase 2: persisted hint ladder + state update + violations -------------
sigp = compute_signals(transcript=[{"role": "assistant", "content": "and then?"}],
                       new_user_message="help", teaching_policy="coached",
                       prior_state={"hint_level": 2, "learner_level": "developing"})
check("prior hint_level surfaced", sigp["hint_level"] == 2)
check("signal block shows the rung", "H2" in build_signal_block(sigp))

st = update_session_state({"hint_level": 1}, {"intervention": "micro_hint"},
                          {"help_requested": True, "turns_without_progress": 2})
check("hint escalates on help", st["hint_level"] >= 2)

st2 = update_session_state({"hint_level": 3}, {"intervention": "probe"},
                           {"help_requested": False, "turns_without_progress": 0, "looks_garbage": False})
check("hint steps DOWN on progress", st2["hint_level"] == 2)

st3 = update_session_state({}, {"intervention": "repair", "mode": "coach"},
                           {"frustration": "high", "recent_probes": 3})
check("repair counted", st3["repairs_done"] == 1 and st3["mode"] == "coach")

st4 = update_session_state({"hint_level": 0}, {"intervention": "reveal", "hint": "5"}, {})
check("explicit tag hint respected", st4["hint_level"] == 5)

check("violation: banned 'isn't specified'",
      "banned_isnt_specified" in detect_violations("That detail isn't specified, so assume a value."))
check("violation: rubber-stamp",
      "possible_rubber_stamp" in detect_violations("That's a solid structure! Well done."))
check("no violation on a clean probe",
      detect_violations("How did you get to 20 million? Take it forward.") == [])

sc = StreamTagStripper()
_ = run_stream(sc, ["<<mode=coach; intervention=micro_hint; hint=1>>\n\nYou're one step away."])
check("stream captured the tag dict", sc.tag.get("mode") == "coach" and sc.tag.get("intervention") == "micro_hint")


print()
if _fail:
    print(f"{len(_fail)} FAILED: {_fail}")
    sys.exit(1)
print("ALL PASS")
