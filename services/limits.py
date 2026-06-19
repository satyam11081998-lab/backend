"""
Canonical free-text input limits (server-authoritative).
Mirror of frontend lib/limits.ts. Recorded in CONTRACTS C4 — changing = announce.

Caps are GENEROUS on purpose: they block automated paste-to-burn-tokens abuse
without ever rejecting a real human answer. `content` carries the candidate's
clarifying questions AND their structured notes/analysis, so its ceiling must be
large enough to never truncate legitimate structure.
"""

ANSWER_MAX_CHARS = 20_000          # one-shot /submit answer (min stays 200)
MESSAGE_MAX_CHARS = 20_000         # per conversational turn (carries structure; min stays 1)
RECOMMENDATION_MAX_CHARS = 20_000  # final recommendation (min stays 20)
