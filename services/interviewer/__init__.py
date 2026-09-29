"""
MECE unified interviewer brain.

One decision system for TEXT, STT and REALTIME VOICE. See
docs/interviewer/MECE_INTERVIEWER_ARCHITECTURE.md.

Public surface used by routes:
    flags.brain_enabled_for(user_id, email)
    engine.decide_turn / word_complete / word_stream / finalize
    dedupe.LEDGER / dedupe.ROWS
    telemetry.emit_timing
"""
