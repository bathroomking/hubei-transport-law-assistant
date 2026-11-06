"""Generate small, renderable PDFs in memory; never commit binary test data."""

from io import BytesIO

from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas


def pdf_bytes(*pages: str, password: str | None = None) -> bytes:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    output = BytesIO()
    encryption = StandardEncryption(password) if password is not None else None
    canvas = Canvas(output, pagesize=(612, 792), encrypt=encryption, invariant=True)
    for text in pages or ("Road construction requires permission.",):
        y = 730
        canvas.setFont("STSong-Light", 12)
        for paragraph in text.splitlines():
            line = ""
            for char in paragraph:
                if pdfmetrics.stringWidth(line + char, "STSong-Light", 12) > 468:
                    canvas.drawString(72, y, line)
                    y -= 18
                    line = ""
                    if y < 60:
                        canvas.showPage()
                        canvas.setFont("STSong-Light", 12)
                        y = 730
                line += char
            if y < 60:
                canvas.showPage()
                canvas.setFont("STSong-Light", 12)
                y = 730
            canvas.drawString(72, y, line)
            y -= 32  # Separate logical paragraphs visibly for layout-aware parsers.
        canvas.showPage()
    canvas.save()
    return output.getvalue()
