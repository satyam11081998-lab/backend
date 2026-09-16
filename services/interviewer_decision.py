"""
Control-tag parsing + streaming strip for the adaptive interviewer (Phase 1).

The adaptive prompt asks the model to prepend a one-line control tag, e.g.
    <<mode=coach; materiality=material; intervention=micro_hint; hint=1>>
which drives state/telemetry and must NEVER reach the candidate. This module
strips it -- from a full string (non-streaming) and from a token STREAM -- and
degrades safely if the model ignores the contract (the reply is never lost).
"""
from __future__ import annotations

import re
from typing import Dict, Generator, Tuple

_TAG_RE = re.compile(r"^\s*<<(.*?)>>\s*", re.DOTALL)


def parse_control_tag(text: str) -> Tuple[Dict[str, str], str]:
    """Return (tag_dict, cleaned_reply). Strips a leading <<...>> tag if present."""
    if not text:
        return {}, ""
    m = _TAG_RE.match(text)
    if not m:
        return {}, text.strip()
    tag: Dict[str, str] = {}
    for part in re.split(r"[;,]", m.group(1)):
        if "=" in part:
            k, v = part.split("=", 1)
            tag[k.strip().lower()] = v.strip().lower()
    cleaned = text[m.end():].lstrip("\n").strip()
    # If the model put its whole reply inside the tag, keep the original text.
    return (tag, cleaned) if cleaned else (tag, text.strip())


class StreamTagStripper:
    """Strip a leading <<...>> control tag from a token STREAM.

    Buffers only until it can decide whether a tag is present, then passes
    tokens through. If the first real chars are not '<<', or no closing '>>'
    arrives within `budget`, it flushes and assumes no tag -- a model that
    ignores the contract never loses its reply.
    """

    def __init__(self, budget: int = 240):
        self.buf = ""
        self.resolved = False
        self.budget = budget
        self._strip_leading = False

    def feed(self, token: str) -> Generator[str, None, None]:
        if self.resolved:
            if token:
                if self._strip_leading:
                    token = token.lstrip()
                    if not token:
                        return
                    self._strip_leading = False
                yield token
            return
        self.buf += token or ""
        stripped = self.buf.lstrip()
        if not stripped:
            return
        if stripped.startswith("<<"):
            end = self.buf.find(">>")
            if end != -1:
                rest = self.buf[end + 2:].lstrip("\n").lstrip()
                self.resolved, self.buf = True, ""
                if rest:
                    yield rest
                else:
                    self._strip_leading = True
            elif len(self.buf) > self.budget:      # never closed -> flush as-is
                out, self.buf, self.resolved = self.buf, "", True
                if out:
                    yield out
            return
        # first real chars are not '<<' -> there is no tag
        if len(stripped) >= 2 or len(self.buf) > 8:
            out, self.buf, self.resolved = self.buf, "", True
            if out:
                yield out

    def flush(self) -> Generator[str, None, None]:
        if not self.buf:
            return
        out, self.buf, self.resolved = self.buf, "", True
        m = _TAG_RE.match(out)          # strip a dangling, unterminated tag
        if m:
            out = out[m.end():].lstrip()
        if out:
            yield out
