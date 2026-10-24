"""Builders for real test files. Parsers are tested against actual PDF and DOCX bytes."""

from __future__ import annotations

import io


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]], title: str | None = None) -> bytes:
    """A minimal valid PDF with one Helvetica text line per list entry."""
    objects: list[bytes] = []
    page_count = len(pages)
    font_id = 3 + 2 * page_count
    info_id = font_id + 1
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(page_count))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode())
    for i, lines in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R"
            f" /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in lines:
            ops.append(f"({_pdf_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Title ({_pdf_escape(title or '')}) >>".encode())

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R"
    if title:
        trailer += f" /Info {info_id} 0 R"
    out.write(f"{trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def make_docx(blocks: list[tuple[str, str]], title: str | None = None) -> bytes:
    """``blocks`` are (style, text): style is "h1", "h2" or "p"."""
    import docx

    document = docx.Document()
    if title:
        document.core_properties.title = title
    for style, text in blocks:
        if style.startswith("h"):
            document.add_heading(text, level=int(style[1]))
        else:
            document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()
