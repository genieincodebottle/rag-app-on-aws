"""Screening text for prompt-injection markers.

This is a filter, not a wall. Pattern matching catches the common, lazy
attacks ("ignore previous instructions") and nothing clever. The controls that
actually limit damage are structural and live elsewhere:

* retrieved text is wrapped in delimited source blocks and the system prompt
  tells the model it is data (``generation/prompts.py``);
* the model has no write tools in standard mode, and agent tools are read-only
  with arguments validated by schema (``agent/tools.py``);
* every citation is checked against the sources the server supplied;
* on AWS, Bedrock Guardrails' prompt-attack filter can screen the question.

Flagged chunks are still stored and retrievable. Deleting them would let an
attacker remove legitimate content by adding a trigger phrase; instead the flag
is shown to the model next to the source and counted in metrics.
"""

from __future__ import annotations

import re

_PATTERNS: dict[str, re.Pattern[str]] = {
    "override_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b"
        r".{0,20}\b(instructions?|prompts?|rules|directions)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "role_reassignment": re.compile(
        r"\b(you are now|act as|pretend to be|from now on you)\b", re.IGNORECASE
    ),
    "prompt_exfiltration": re.compile(
        r"\b(reveal|print|repeat|show|output)\b.{0,30}\b(system prompt|hidden instructions|"
        r"your instructions|api key|secret)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    "fake_delimiter": re.compile(
        r"</?(source|sources|system|instructions?|question)\b|\[/?INST\]|<\|im_(start|end)\|>",
        re.IGNORECASE,
    ),
    "exfil_link": re.compile(r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*=", re.IGNORECASE),
}


def screen_text(text: str) -> list[str]:
    """Return the names of the patterns found in ``text``."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]
