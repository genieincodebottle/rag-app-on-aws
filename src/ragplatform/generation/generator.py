"""Grounded answer generation: prompt, call, parse, validate, optional grounding check."""

from __future__ import annotations

from collections.abc import Sequence

from ragplatform.config import Settings
from ragplatform.domain.errors import UpstreamError
from ragplatform.domain.models import Answer, AnswerStatus, RetrievedChunk, TokenUsage
from ragplatform.generation.answers import (
    AnswerFormatError,
    downgrade_ungrounded,
    finalize_answer,
    parse_raw_answer,
)
from ragplatform.generation.prompts import (
    ANSWER_SYSTEM,
    FORMAT_REMINDER,
    answer_prompt,
    source_blocks,
)
from ragplatform.observability.telemetry import Telemetry, logger, tracer
from ragplatform.providers.base import ChatModel, ChatRequest
from ragplatform.providers.factory import GroundingCheck


class Generator:
    def __init__(
        self,
        *,
        chat: ChatModel,
        grounding: GroundingCheck | None,
        settings: Settings,
        telemetry: Telemetry,
    ) -> None:
        self.chat = chat
        self.grounding = grounding
        self.settings = settings
        self.telemetry = telemetry

    @tracer.capture_method(capture_response=False)
    def answer(
        self, question: str, sources: Sequence[RetrievedChunk], snippets: dict[str, str]
    ) -> Answer:
        blocks = source_blocks(sources)
        prompt = answer_prompt(question, blocks)
        usage = TokenUsage()
        last_error: AnswerFormatError | None = None
        # One retry with a format reminder. A second malformed reply is an
        # upstream error, not something to paper over with a default answer.
        for attempt in range(2):
            result = self.chat.generate(
                ChatRequest(
                    system=ANSWER_SYSTEM,
                    prompt=prompt + (FORMAT_REMINDER if attempt else ""),
                    max_tokens=self.settings.max_output_tokens,
                    temperature=self.settings.temperature,
                    purpose="answer",
                    question=question,
                    sources=blocks,
                )
            )
            usage = usage + result.usage
            self.telemetry.add("ModelLatency", result.latency_ms, "ms")
            try:
                raw = parse_raw_answer(result.text)
                break
            except AnswerFormatError as exc:
                last_error = exc
                self.telemetry.add("AnswerFormatRetry", 1)
                logger.warning(
                    "model output not valid JSON",
                    extra={"attempt": attempt + 1, "stop_reason": result.stop_reason},
                )
        else:
            raise UpstreamError(f"model returned malformed output twice: {last_error}")

        answer = finalize_answer(
            raw, sources=sources, snippets=snippets, model_id=result.model_id, usage=usage
        )
        if answer.invalid_citations_removed:
            self.telemetry.add("InvalidCitationsRemoved", answer.invalid_citations_removed)

        if self.grounding is not None and answer.status == AnswerStatus.ANSWERED:
            context = "\n\n".join(s.text for s in sources)
            if not self.grounding.is_grounded(
                sources=context, question=question, answer=answer.answer
            ):
                self.telemetry.add("GroundingCheckFailed", 1)
                answer = downgrade_ungrounded(answer)
        return answer
