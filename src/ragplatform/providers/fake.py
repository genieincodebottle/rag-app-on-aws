"""Deterministic offline providers for tests, CI and keyless local runs.

``HashingEmbedder`` maps words and word pairs into a fixed-size vector with a
stable hash. Texts that share words land near each other, which is enough to
exercise vector search, fusion and filters end to end. It has no notion of
synonyms, so retrieval quality numbers measured with it say nothing about a
real embedding model.

``ExtractiveChatModel`` answers by copying the source sentences that best
overlap the question, cites them with their markers, and reports insufficient
context when nothing overlaps enough. It never follows instructions found in
sources, which makes it a useful control in injection tests: any change in
behaviour comes from the pipeline, not from the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from itertools import pairwise

from ragplatform.domain.models import TokenUsage
from ragplatform.providers.base import ChatRequest, ChatResult, EmbeddingSpace, l2_normalize

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "my",
        "your",
        "our",
        "their",
        "there",
        "any",
        "all",
        "some",
        "no",
        "not",
        "if",
        "then",
        "than",
        "so",
        "such",
        "into",
        "about",
        "over",
        "under",
        "per",
        "each",
    ]
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _stem(word: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def terms(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def _bucket(feature: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    def __init__(self, dimensions: int) -> None:
        self.space = EmbeddingSpace("fake", "hashing-v1", dimensions)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.space.dimensions
        words = terms(text)
        features = [(w, 1.0) for w in words] + [(f"{a} {b}", 0.5) for a, b in pairwise(words)]
        if not features:
            features = [("<empty>", 1.0)]
        for feature, weight in features:
            index, sign = _bucket(feature, self.space.dimensions)
            vector[index] += sign * weight
        return l2_normalize(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OverlapReranker:
    model_id = "fake-overlap-reranker"

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        wanted = set(terms(query))
        scored = [
            (i, len(wanted & set(terms(doc))) / (len(wanted) or 1))
            for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_n]


class ExtractiveChatModel:
    model_id = "fake-extractive-v1"

    def __init__(self, min_overlap: float = 0.5) -> None:
        self.min_overlap = min_overlap

    def generate(self, request: ChatRequest) -> ChatResult:
        if request.purpose == "plan":
            text = json.dumps(self._plan(request.question))
        else:
            text = json.dumps(self._answer(request))
        usage = TokenUsage(
            input_tokens=(len(request.system) + len(request.prompt)) // 4,
            output_tokens=max(1, len(text) // 4),
        )
        return ChatResult(text=text, usage=usage, model_id=self.model_id)

    def _plan(self, question: str) -> dict[str, object]:
        parts = [
            p.strip(" ?.")
            for p in re.split(r"\?\s+|\s+and also\s+|;\s+", question)
            if p.strip(" ?.")
        ]
        if len(parts) >= 2 and all(len(terms(p)) >= 2 for p in parts):
            return {"strategy": "decompose", "sub_questions": [f"{p}?" for p in parts[:3]]}
        return {"strategy": "direct", "sub_questions": []}

    def _answer(self, request: ChatRequest) -> dict[str, object]:
        wanted = set(terms(request.question))
        candidates: list[tuple[float, str, str]] = []
        for source in request.sources:
            for sentence in _SENTENCE.split(source.text):
                sentence = sentence.strip()
                have = set(terms(sentence))
                if not sentence or not have or not wanted:
                    continue
                score = len(wanted & have) / len(wanted)
                candidates.append((score, source.marker, sentence))
        candidates.sort(key=lambda c: -c[0])
        best = [c for c in candidates if c[0] >= self.min_overlap]
        if not best:
            return {
                "status": "insufficient_context",
                "answer": "",
                "citations": [],
                "conflicts": [],
            }

        top_score, top_marker, top_sentence = best[0]
        chosen = [(top_marker, top_sentence)]
        for score, marker, sentence in best[1:]:
            if marker != top_marker and score >= top_score * 0.9:
                chosen.append((marker, sentence))
                break

        if len(chosen) == 2:
            first_numbers = set(_NUMBER.findall(chosen[0][1]))
            second_numbers = set(_NUMBER.findall(chosen[1][1]))
            if first_numbers and second_numbers and first_numbers != second_numbers:
                return {
                    "status": "conflicting_sources",
                    "answer": f"{chosen[0][1]} [{chosen[0][0]}] However, {chosen[1][1]} "
                    f"[{chosen[1][0]}]",
                    "citations": [chosen[0][0], chosen[1][0]],
                    "conflicts": [f"{chosen[0][0]} and {chosen[1][0]} give different values."],
                }
        answer = " ".join(f"{sentence} [{marker}]" for marker, sentence in chosen)
        return {
            "status": "answered",
            "answer": answer,
            "citations": [marker for marker, _ in chosen],
            "conflicts": [],
        }
