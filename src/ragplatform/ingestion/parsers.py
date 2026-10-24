"""File type detection and parsing into heading-aware sections.

Each parser returns ``Section`` objects that carry the heading path and page
range, which later become the citation location shown to the user. Parsers
reject what they cannot handle with a ``PermanentIngestError``, so a bad file
is marked failed once instead of being retried until it reaches the DLQ.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import ClassVar

from ragplatform.domain.errors import PermanentIngestError
from ragplatform.domain.models import ParsedDocument, Section
from ragplatform.ingestion.normalize import normalize_text

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/html": "html",
    "text/markdown": "markdown",
    "text/plain": "text",
    "text/csv": "csv",
}

EXTENSION_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".csv": "text/csv",
}


@dataclass(frozen=True)
class ParseLimits:
    max_pdf_pages: int
    max_docx_uncompressed_bytes: int


def sniff_family(data: bytes) -> str:
    """Identify the container from the bytes, independent of the declared type."""
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        return "zip"
    try:
        data[:65536].decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the 64 KiB boundary is still text.
        if exc.start < len(data[:65536]) - 4:
            return "binary"
    return "text"


def check_declared_type(data: bytes, content_type: str) -> str:
    kind = SUPPORTED_TYPES.get(content_type)
    if kind is None:
        raise PermanentIngestError(
            "unsupported_type", f"content type {content_type} is not supported"
        )
    family = sniff_family(data)
    expected = {"pdf": "pdf", "docx": "zip"}.get(kind, "text")
    if family != expected:
        raise PermanentIngestError(
            "type_mismatch",
            f"declared {content_type} but the file content looks like {family}",
        )
    return kind


def parse(data: bytes, content_type: str, limits: ParseLimits) -> ParsedDocument:
    kind = check_declared_type(data, content_type)
    parsers: dict[str, Callable[[bytes, ParseLimits], ParsedDocument]] = {
        "pdf": _parse_pdf,
        "docx": _parse_docx,
        "html": lambda d, _l: _parse_html(_decode(d)),
        "markdown": lambda d, _l: _parse_markdown(_decode(d)),
        "text": lambda d, _l: _single_section(_decode(d)),
        "csv": lambda d, _l: _parse_csv(_decode(d)),
    }
    parsed = parsers[kind](data, limits)
    sections = tuple(
        Section(
            text=normalize_text(s.text),
            heading_path=s.heading_path,
            page_start=s.page_start,
            page_end=s.page_end,
        )
        for s in parsed.sections
    )
    sections = tuple(s for s in sections if s.text)
    if not sections:
        raise PermanentIngestError(
            "no_text", "no extractable text (scanned PDFs need OCR, which is not enabled)"
        )
    return ParsedDocument(title=parsed.title, sections=sections, page_count=parsed.page_count)


def _decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def _single_section(text: str) -> ParsedDocument:
    return ParsedDocument(title=None, sections=(Section(text=text),))


# --------------------------------------------------------------------------- PDF


def _parse_pdf(data: bytes, limits: ParseLimits) -> ParsedDocument:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PermanentIngestError("encrypted", "encrypted PDFs are not supported")
        pages = reader.pages
        if len(pages) > limits.max_pdf_pages:
            raise PermanentIngestError(
                "too_many_pages", f"{len(pages)} pages exceeds the limit of {limits.max_pdf_pages}"
            )
        sections = [
            Section(text=page.extract_text() or "", page_start=i, page_end=i)
            for i, page in enumerate(pages, start=1)
        ]
        title = None
        if reader.metadata is not None and reader.metadata.title:
            title = str(reader.metadata.title)
    except PdfReadError as exc:
        raise PermanentIngestError("corrupt", f"the PDF could not be read: {exc}") from exc
    return ParsedDocument(title=title, sections=tuple(sections), page_count=len(sections))


# --------------------------------------------------------------------------- DOCX


def _parse_docx(data: bytes, limits: ParseLimits) -> ParsedDocument:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if "word/document.xml" not in names:
                raise PermanentIngestError("type_mismatch", "the zip file is not a DOCX document")
            # Guard against decompression bombs before python-docx inflates anything.
            total = sum(info.file_size for info in archive.infolist())
            if total > limits.max_docx_uncompressed_bytes:
                raise PermanentIngestError(
                    "decompression_limit", f"uncompressed size {total} bytes exceeds the limit"
                )
    except zipfile.BadZipFile as exc:
        raise PermanentIngestError("corrupt", "the DOCX file is not a valid zip archive") from exc

    import docx

    document = docx.Document(io.BytesIO(data))
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section(text="\n".join(buffer), heading_path=tuple(path)))
            buffer.clear()

    for paragraph in document.paragraphs:
        style = (paragraph.style.name if paragraph.style is not None else "") or ""
        match = re.match(r"Heading (\d)", style)
        text = paragraph.text.strip()
        if match and text:
            flush()
            level = int(match.group(1))
            del path[level - 1 :]
            path.append(text)
        elif text:
            buffer.append(text)
    flush()
    for table in document.tables:
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        if rows:
            sections.append(Section(text="\n".join(rows), heading_path=("Table",)))
    title = document.core_properties.title or None
    return ParsedDocument(title=title, sections=tuple(sections))


# --------------------------------------------------------------------------- HTML


class _HTMLSections(HTMLParser):
    _SKIP: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "template", "svg"})
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "li", "tr", "br", "section", "article", "table", "ul", "ol", "pre"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.path: list[str] = []
        self.title: str | None = None
        self._buffer: list[str] = []
        self._skip_stack: list[str] = []
        self._heading_level: int | None = None
        self._heading_text: list[str] = []
        self._in_title = False

    _VOID: ClassVar[frozenset[str]] = frozenset(
        {"br", "img", "input", "meta", "link", "hr", "area", "base", "col", "source", "wbr"}
    )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        style = (attributes.get("style") or "").replace(" ", "").lower()
        hidden = "hidden" in attributes or "display:none" in style or "visibility:hidden" in style
        if self._skip_stack or tag in self._SKIP or hidden:
            # Text a browser would not show is dropped: it is a common place to
            # hide instructions aimed at the model.
            if tag not in self._VOID:
                self._skip_stack.append(tag)
            return
        if tag == "title":
            self._in_title = True
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._heading_level = int(tag[1])
            self._heading_text = []
        elif tag in self._BLOCK:
            self._buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_stack:
            if tag in self._skip_stack:
                # Pop to the matching tag, tolerating unclosed inner elements.
                while self._skip_stack and self._skip_stack.pop() != tag:
                    pass
            return
        if tag == "title":
            self._in_title = False
        if self._heading_level is not None and tag == f"h{self._heading_level}":
            heading = " ".join("".join(self._heading_text).split())
            if heading:
                del self.path[self._heading_level - 1 :]
                self.path.append(heading)
            self._heading_level = None

    def handle_data(self, data: str) -> None:
        if self._skip_stack:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
        elif self._heading_level is not None:
            self._heading_text.append(data)
        else:
            self._buffer.append(data)

    def _flush(self) -> None:
        text = "".join(self._buffer).strip()
        if text:
            self.sections.append(Section(text=text, heading_path=tuple(self.path)))
        self._buffer = []

    def close(self) -> None:
        super().close()
        self._flush()


def _parse_html(text: str) -> ParsedDocument:
    parser = _HTMLSections()
    parser.feed(text)
    parser.close()
    return ParsedDocument(title=parser.title or None, sections=tuple(parser.sections))


# --------------------------------------------------------------------------- Markdown

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _parse_markdown(text: str) -> ParsedDocument:
    sections: list[Section] = []
    path: list[str] = []
    buffer: list[str] = []
    title: str | None = None
    in_fence = False

    def flush() -> None:
        if "".join(buffer).strip():
            sections.append(Section(text="\n".join(buffer), heading_path=tuple(path)))
        buffer.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else _ATX.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2)
            if level == 1 and title is None:
                title = heading
            del path[level - 1 :]
            path.append(heading)
        else:
            buffer.append(line)
    flush()
    return ParsedDocument(title=title, sections=tuple(sections))


# --------------------------------------------------------------------------- CSV

_CSV_ROWS_PER_SECTION = 25


def _parse_csv(text: str) -> ParsedDocument:
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return ParsedDocument(title=None, sections=())
    header, body = rows[0], rows[1:]
    sections = []
    for start in range(0, len(body), _CSV_ROWS_PER_SECTION):
        # Each row is written as "column: value" pairs so a chunk is readable
        # without its header row, which may sit in a different chunk.
        lines = [
            "; ".join(f"{h}: {v}" for h, v in zip(header, row, strict=False) if v)
            for row in body[start : start + _CSV_ROWS_PER_SECTION]
        ]
        sections.append(
            Section(text="\n".join(lines), heading_path=(f"Rows {start + 1}-{start + len(lines)}",))
        )
    return ParsedDocument(title=None, sections=tuple(sections))
