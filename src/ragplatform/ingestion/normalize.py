"""Text normalisation applied to every parsed section before chunking."""

from __future__ import annotations

import re
import unicodedata

# Zero-width and bidirectional control characters render as nothing but survive
# into embeddings and prompts. They are a common way to hide instructions in a
# document, so they are removed rather than preserved.
_INVISIBLE = re.compile("[\u00ad\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # "infor-\nmation" from PDF line wraps
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def had_invisible_characters(text: str) -> bool:
    return bool(_INVISIBLE.search(text))
