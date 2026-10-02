"""Small, dependency-free text helpers shared by several engines."""

from __future__ import annotations

import re
from typing import Iterable, List, Set

_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from", "as", "is", "are",
    "was", "were", "be", "been", "it", "that", "this", "your", "you", "i", "we", "our", "my", "me", "how", "what",
    "would", "do", "did", "does", "can", "could", "about", "tell", "time", "when", "which", "why", "us",
}
_TOKEN = re.compile(r"[a-z0-9%₹$]+")


def tokens(text: str, *, keep_stop: bool = False) -> List[str]:
    toks = _TOKEN.findall((text or "").lower())
    return toks if keep_stop else [t for t in toks if t not in _STOP]


def token_set(text: str) -> Set[str]:
    return set(tokens(text))


def jaccard(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(needle: str, haystack: str) -> float:
    """Share of the needle's tokens (stopwords kept) that appear, in order-insensitive fashion,
    in the haystack. Used to verify that an evidence quote really came from the candidate."""
    n = tokens(needle, keep_stop=True)
    if not n:
        return 0.0
    hay = tokens(haystack, keep_stop=True)
    if not hay:
        return 0.0
    # sliding-window best match keeps it robust to small transcription differences
    hs = set(hay)
    simple = sum(1 for t in n if t in hs) / len(n)
    if simple < 0.6:
        return simple
    w = len(n)
    best = 0.0
    for i in range(0, max(1, len(hay) - w + 1)):
        window = set(hay[i:i + w + 3])
        score = sum(1 for t in n if t in window) / len(n)
        if score > best:
            best = score
            if best >= 0.999:
                break
    return best


def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ɏ']+|\d+(?:[.,]\d+)*%?", text or "")


def truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def any_in(text: str, phrases: Iterable[str]) -> bool:
    low = (text or "").lower()
    return any(p in low for p in phrases)
