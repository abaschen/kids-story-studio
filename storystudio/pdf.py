"""The finished book as a PDF: reading order (A5 portrait pages) or a print-ready booklet (A4 landscape sheets)."""

from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, A5, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

W, H = A5
MARGIN = 28
INK, PAPER = HexColor("#2d2a32"), HexColor("#fffaf2")


def _img(path: Path) -> ImageReader:
    """JPEG keeps a 20-page book around 5 MB instead of 50."""
    buf = BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=90)
    buf.seek(0)
    return ImageReader(buf)


def _text(
    c: Canvas, text: str, x: float, top: float, width: float, height: float, size: float, *, bold: bool = False
) -> None:
    """Draw wrapped text, shrinking the font until it fits the box."""
    while True:
        style = ParagraphStyle(
            "t",
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            alignment=1 if bold else 0,
            leading=size * 1.35,
            textColor=INK,
        )
        para = Paragraph(escape(text).replace("\n", "<br/>"), style)
        _, h = para.wrap(width, height)
        if h <= height or size <= 8:
            para.drawOn(c, x, top - h)
            return
        size -= 1


def _page(c: Canvas, story: dict[str, Any], root: Path, item: dict[str, Any] | None) -> None:
    """One A5 page at the canvas origin: a picture on top, the read-aloud text below. item=None is a blank page."""
    if item is None:
        return
    c.setFillColor(PAPER)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(INK)
    inner = W - 2 * MARGIN
    if item["kind"] == "cover":
        _text(c, story["title"], MARGIN, H - 60, inner, 120, 30, bold=True)
        cover = next((p["image"] for p in story["pages"] if p.get("image")), None)
        if cover:
            c.drawImage(_img(root / cover), MARGIN, 90, inner, inner * 0.75)
    elif item["kind"] == "end":
        c.setFont("Helvetica-Oblique", 16)
        c.drawCentredString(W / 2, H / 2, "The end")
    else:
        p = item["page"]
        pic_h = inner * 0.75
        if p.get("image"):
            c.drawImage(_img(root / p["image"]), MARGIN, H - MARGIN - pic_h, inner, pic_h)
        _text(c, p["text"], MARGIN, H - MARGIN - pic_h - 18, inner, H - 2 * MARGIN - pic_h - 40, 18)
        c.setFont("Helvetica", 8)
        c.drawCentredString(W / 2, 14, str(p["n"]))


def _items(story: dict[str, Any]) -> list[dict[str, Any] | None]:
    return [{"kind": "cover"}, *({"kind": "page", "page": p} for p in story["pages"]), {"kind": "end"}]


def build(story: dict[str, Any], root: Path, layout: str = "pages") -> bytes:
    """layout "pages": one A5 page per PDF page. "booklet": A4 landscape sheets, two A5 pages each, ordered so that
    printing double-sided (flip on the short edge), folding the stack in half and stapling the fold reads in order."""
    items = _items(story)
    buf = BytesIO()
    if layout == "pages":
        c = Canvas(buf, pagesize=A5, pageCompression=1)
        for it in items:
            _page(c, story, root, it)
            c.showPage()
    else:
        items += [None] * (-len(items) % 4)
        n = len(items)
        c = Canvas(buf, pagesize=landscape(A4), pageCompression=1)
        for i in range(n // 4):
            # Front of sheet i: last-2i | first+2i; back: first+2i+1 | last-2i-1 (0-based).
            for left, right in ((n - 1 - 2 * i, 2 * i), (2 * i + 1, n - 2 - 2 * i)):
                for slot, idx in enumerate((left, right)):
                    c.saveState()
                    c.translate(slot * W, 0)
                    _page(c, story, root, items[idx])
                    c.restoreState()
                c.setStrokeColor(HexColor("#cccccc"))
                c.setDash(3, 3)
                c.line(W, 0, W, H)  # fold line
                c.showPage()
    c.setTitle(story["title"])
    c.save()
    return buf.getvalue()
