"""Chunking strategies.

``recursive``
    Split each section on paragraph, line, sentence, then word boundaries until
    every piece fits ``chunk_tokens``, with ``chunk_overlap_tokens`` of overlap.
    Chunks never cross a section boundary, so a chunk never mixes two headings.

``parent_child``
    Split each section into parents of up to ``parent_chunk_tokens``, then split
    each parent into children of ``chunk_tokens``. Children are embedded and
    searched; the parent is what the model reads. Small chunks match a question
    precisely, and the larger parent gives the model enough surrounding text.

With ``contextual_headers`` the text sent to the embedder is prefixed with the
document title and section path ("Expense Policy > Travel > Hotels"). This is a
no-LLM version of contextual retrieval: a chunk that says "the limit is 200"
becomes findable by a question about hotel limits. The stored chunk text, and
so the text shown in citations, is unchanged.

Token counts are estimated as characters / 4. This is close enough for budget
decisions in English text and needs no tokenizer that matches the model.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from itertools import pairwise

from ragplatform.config import ChunkStrategy
from ragplatform.domain.models import ChunkDraft, ChunkKind, ParsedDocument, Section
from ragplatform.security.injection import screen_text

_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).lower().encode()).hexdigest()


def split_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Recursive split into pieces of at most ``max_tokens`` with overlap."""
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    pieces = _split(text.strip(), max_chars, _SEPARATORS)
    if overlap_chars <= 0 or len(pieces) < 2:
        return pieces
    with_overlap = [pieces[0]]
    for previous, piece in pairwise(pieces):
        tail = previous[-overlap_chars:]
        # Start the overlap on a word boundary so no chunk begins mid-word.
        space = tail.find(" ")
        tail = tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
        candidate = f"{tail} {piece}".strip()
        with_overlap.append(candidate if len(candidate) <= max_chars + overlap_chars else piece)
    return with_overlap


def _split(text: str, max_chars: int, separators: Sequence[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    separator, rest = separators[0], separators[1:]
    parts = text.split(separator)
    if len(parts) == 1:
        return _split(text, max_chars, rest)
    chunks: list[str] = []
    current = ""
    for part in parts:
        joined = f"{current}{separator}{part}" if current else part
        if len(joined) <= max_chars:
            current = joined
            continue
        if current:
            chunks.append(current.strip())
        if len(part) > max_chars:
            chunks.extend(_split(part, max_chars, rest))
            current = ""
        else:
            current = part
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _section_path(section: Section) -> str:
    return " > ".join(section.heading_path)


def embed_text_for(title: str, section_path: str, text: str, contextual: bool) -> str:
    if not contextual:
        return text
    header = " > ".join(part for part in (title, section_path) if part)
    return f"{header}\n\n{text}" if header else text


def build_chunks(
    parsed: ParsedDocument,
    *,
    title: str,
    strategy: ChunkStrategy,
    chunk_tokens: int,
    overlap_tokens: int,
    parent_tokens: int,
    contextual_headers: bool,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    seen_hashes: set[str] = set()

    def add(
        kind: ChunkKind, text: str, section: Section, parent_index: int | None = None
    ) -> int | None:
        text_digest = content_hash(text)
        # Repeated boilerplate (page headers, legal footers) is stored once.
        # Parents take no part: dropping one would orphan its children, and a
        # short section's only child has exactly its parent's text.
        if kind != ChunkKind.PARENT:
            if text_digest in seen_hashes:
                return None
            seen_hashes.add(text_digest)
        path = _section_path(section)
        embed_text = (
            None
            if kind == ChunkKind.PARENT
            else embed_text_for(title, path, text, contextual_headers)
        )
        index = len(drafts)
        drafts.append(
            ChunkDraft(
                chunk_index=index,
                kind=kind,
                text=text,
                embed_text=embed_text,
                section_path=path,
                page_start=section.page_start,
                page_end=section.page_end,
                # Keyed on the embedded text, so a stored vector is reused only
                # when the embedder would see exactly the same input.
                content_hash=content_hash(embed_text) if embed_text is not None else text_digest,
                token_count=estimate_tokens(text),
                parent_index=parent_index,
                risk_flags=tuple(screen_text(text)),
            )
        )
        return index

    for section in _merge_small_sections(parsed.sections, chunk_tokens):
        if strategy == ChunkStrategy.RECURSIVE:
            for piece in split_text(section.text, chunk_tokens, overlap_tokens):
                add(ChunkKind.FLAT, piece, section)
        else:
            for parent_text in split_text(section.text, parent_tokens, 0):
                parent_index = add(ChunkKind.PARENT, parent_text, section)
                for child in split_text(parent_text, chunk_tokens, overlap_tokens):
                    add(ChunkKind.CHILD, child, section, parent_index)
    return drafts


def _merge_small_sections(sections: Iterable[Section], chunk_tokens: int) -> list[Section]:
    """Merge consecutive tiny sections that share a heading path.

    PDFs produce one section per page and some pages hold a single line. Left
    alone, those become chunks too small to carry meaning.
    """
    merged: list[Section] = []
    for section in sections:
        if (
            merged
            and merged[-1].heading_path == section.heading_path
            and estimate_tokens(merged[-1].text) < chunk_tokens // 3
        ):
            last = merged.pop()
            section = Section(
                text=f"{last.text}\n\n{section.text}",
                heading_path=section.heading_path,
                page_start=last.page_start if last.page_start is not None else section.page_start,
                page_end=section.page_end if section.page_end is not None else last.page_end,
            )
        merged.append(section)
    return merged


_SLUG = re.compile(r"[^a-z0-9]+")


def default_title(source_name: str) -> str:
    stem = source_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return _SLUG.sub(" ", stem.lower()).strip().title() or "Untitled"
