"""
Text normalisation for classification only.

The candidate's original words are what gets persisted and shown to models; this
normalised form exists only so deterministic detectors are robust to casing,
curly quotes, chat shorthand ("pls", "u"), a few common Hinglish phrases and
speech-transcription noise. It never changes numbers.
"""
from __future__ import annotations

import re
import unicodedata

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    " ": " ",
}

# Chat shorthand / frequent typos -> canonical words. Whole-word replacements only.
_SHORTHAND = {
    # Deliberately NOT mapped: "k" (10 k users), "r" (R&D), "y" (x and y), "id" (user id).
    "pls": "please", "plz": "please", "plss": "please", "pleasee": "please", "pleas": "please",
    "u": "you", "ur": "your", "kk": "ok", "okk": "ok",
    "okie": "ok", "okay": "ok", "thx": "thanks", "ty": "thanks",
    "dont": "don't", "cant": "can't", "wont": "won't", "im": "i'm", "ive": "i've",
    "ill": "i'll", "isnt": "isn't", "doesnt": "doesn't", "didnt": "didn't",
    "whats": "what's", "thats": "that's", "lets": "let's", "hows": "how's", "wheres": "where's",
    "hlp": "help", "halp": "help", "hepl": "help", "hint?": "hint",
    "gimme": "give me", "wanna": "want to", "gonna": "going to", "dunno": "don't know",
    "idk": "i don't know", "nvm": "never mind", "sry": "sorry", "tho": "though",
    "abt": "about", "approx": "approximately", "b/w": "between", "w/o": "without",
    "ans": "answer", "soln": "solution", "sol": "solution",
    "wat": "what", "wht": "what", "wot": "what", "hw": "how", "shud": "should", "shld": "should",
    "cud": "could", "wud": "would", "cn": "can", "knw": "know", "kno": "know", "nd": "and", "thnk": "think",
    "plz.": "please", "becuz": "because", "bcoz": "because", "bcz": "because", "coz": "because", "cuz": "because",
}

# Hinglish phrases mapped to the English intent they carry. Order matters (longest first).
_HINGLISH = [
    (r"\bkuch (bhi )?samajh nahi aa raha\b", "i don't understand"),
    (r"\bsamajh nahi aa raha\b", "i don't understand"),
    (r"\bsamajh nahi aaya\b", "i don't understand"),
    (r"\bsamajh (gaya|gayi|aa gaya)\b", "got it"),
    (r"\bmadad (karo|kariye|chahiye)\b", "help me"),
    (r"\bmadad\b", "help"),
    (r"\bhint (do|dedo|de do|dijiye|batao)\b", "give me a hint"),
    (r"\b(answer|solution|jawab) (batao|bata do|dedo|de do|dikhao)\b", "tell me the answer"),
    (r"\bkaise (karu|karun|karein|start karu|shuru karu)\b", "how do i start"),
    (r"\bkya karu\b", "what should i do"),
    (r"\bpata nahi\b", "i don't know"),
    (r"\bmujhe nahi pata\b", "i don't know"),
    (r"\bbakwas\b", "this is useless"),
    (r"\bachha\b|\bacha\b", "ok"),
    (r"\bhaan\b|\bhan ji\b", "yes"),
    (r"\bnahi\b", "no"),
    (r"\byaar\b|\bbhai\b|\bbro\b", ""),
]

_ASR_FILLERS_ONLY = {
    "", "you", "thank you", "thanks", "thanks for watching", "thank you for watching", "bye",
    "okay", "ok", "uh", "um", "hmm", "mm", "mhm", "so", ".", "...", "yeah", "the", "a",
}


def base_clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    for a, b in _QUOTES.items():
        t = t.replace(a, b)
    return t


def normalize(text: str) -> str:
    """Lowercased, shorthand-expanded, whitespace-collapsed. Numbers untouched."""
    t = base_clean(text).lower()
    for pat, rep in _HINGLISH:
        t = re.sub(pat, rep, t)
    # Collapse elongated letters ("pleaseeee", "helppp", "sooo") to at most two.
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)
    words = re.split(r"(\s+)", t)
    out = []
    for w in words:
        core = w.strip(",.!?;:")
        if core in _SHORTHAND:
            prefix = w[: len(w) - len(w.lstrip(",.!?;:"))]
            suffix = w[len(w.rstrip(",.!?;:")):]
            w = prefix + _SHORTHAND[core] + suffix
        out.append(w)
    t = "".join(out)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s%.$₹]", " ", text or "").strip()


def words(text: str) -> list:
    return re.findall(r"[a-z0-9₹$%.']+", (text or "").lower())


def is_asr_filler_only(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", base_clean(text).lower()).strip()
    t = re.sub(r"\s+", " ", t)
    return t in _ASR_FILLERS_ONLY


def dedupe_repeated_words(text: str) -> str:
    """'wait wait wait' -> 'wait'; 'the the' -> 'the'. For classification only."""
    return re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text or "", flags=re.IGNORECASE)


def ends_with_question(text: str) -> bool:
    return (text or "").rstrip().rstrip('"\')').endswith("?")
