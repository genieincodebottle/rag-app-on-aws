"""Turning model output into a validated ``Answer``.

The model's JSON is parsed strictly, then checked against what the server
actually sent:

* every ``[S#]`` marker must name a source that was in the prompt; unknown
  markers are removed and counted;
* an "answered" result with no valid citation is downgraded to
  ``insufficient_context``, because an uncited answer cannot be traced to a
  document and is the shape a hallucination takes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ragplatform.domain.models import Answer, AnswerStatus, Citation, RetrievedChunk, TokenUsage

INSUFFICIENT_MESSAGE = (
    "The documents you can access do not contain enough information to answer this question."
)

_MARKER_GROUP = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class AnswerFormatError(ValueError):
    pass


class RawAnswer(BaseModel):
    status: Literal["answered", "insufficient_context", "conflicting_sources"]
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


def parse_raw_answer(text: str) -> RawAnswer:
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise AnswerFormatError("no JSON object in model output")
    try:
        return RawAnswer.model_validate(json.loads(cleaned[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise AnswerFormatError(str(exc)) from exc


def _snippet(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."


def finalize_answer(
    raw: RawAnswer,
    *,
    sources: Sequence[RetrievedChunk],
    snippets: dict[str, str],
    model_id: str,
    usage: TokenUsage,
) -> Answer:
    known = {f"S{i}": chunk for i, chunk in enumerate(sources, start=1)}
    removed = 0
    order: list[str] = []

    def keep_valid(match: re.Match[str]) -> str:
        nonlocal removed
        markers = [m.strip() for m in match.group(1).split(",")]
        valid = [m for m in markers if m in known]
        removed += len(markers) - len(valid)
        for marker in valid:
            if marker not in order:
                order.append(marker)
        return "[" + ", ".join(valid) + "]" if valid else ""

    text = _MARKER_GROUP.sub(keep_valid, raw.answer).strip()
    text = re.sub(r"\s+([.,;])", r"\1", re.sub(r"[ \t]{2,}", " ", text))
    for marker in raw.citations:
        if marker in known and marker not in order:
            order.append(marker)
        elif marker not in known:
            removed += 1

    status = AnswerStatus(raw.status)
    downgraded: str | None = None
    if status != AnswerStatus.INSUFFICIENT_CONTEXT and not order:
        status, downgraded = AnswerStatus.INSUFFICIENT_CONTEXT, "no_valid_citations"
    if status == AnswerStatus.INSUFFICIENT_CONTEXT:
        return Answer(
            status=status,
            answer=INSUFFICIENT_MESSAGE,
            citations=[],
            conflicts=[],
            model_id=model_id,
            usage=usage,
            invalid_citations_removed=removed,
            downgraded_reason=downgraded,
        )

    citations = []
    for marker in order:
        chunk = known[marker]
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version=chunk.version,
                title=chunk.title,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                snippet=_snippet(snippets.get(str(chunk.chunk_id), chunk.text)),
            )
        )
    return Answer(
        status=status,
        answer=text,
        citations=citations,
        conflicts=raw.conflicts if status == AnswerStatus.CONFLICTING_SOURCES else [],
        model_id=model_id,
        usage=usage,
        invalid_citations_removed=removed,
        downgraded_reason=downgraded,
    )


def downgrade_ungrounded(answer: Answer) -> Answer:
    return answer.model_copy(
        update={
            "status": AnswerStatus.INSUFFICIENT_CONTEXT,
            "answer": INSUFFICIENT_MESSAGE,
            "citations": [],
            "conflicts": [],
            "downgraded_reason": "grounding_check_failed",
        }
    )
