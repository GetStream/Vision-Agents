# /// script
# requires-python = ">=3.11"
# dependencies = ["reportlab==4.2.5", "pypdf==5.1.0", "openpyxl==3.1.5", "pillow==11.0.0", "msoffcrypto-tool==5.4.2"]
# ///
"""Writes the files the extract tests read.

Run from this directory with `uv run generate.py`. It needs pandoc on the PATH, and
macOS's textutil for the legacy Word file, since nothing else here writes one.
"""

import io
import subprocess
from pathlib import Path

import openpyxl
from msoffcrypto.format.ooxml import OOXMLFile
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = Path(__file__).parent
WIDTH, HEIGHT = A4


def manual_pdf() -> None:
    """Twelve pages with a running header and footer, headings, a list and a table."""
    pdf = canvas.Canvas(str(HERE / "manual.pdf"), pagesize=A4)
    for page in range(1, 13):
        pdf.setFont("Helvetica", 8)
        pdf.drawString(20 * mm, HEIGHT - 12 * mm, "Equipment Manual - Kitchen Operations")
        pdf.drawString(WIDTH / 2, 10 * mm, str(page))
        y = HEIGHT - 30 * mm
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(20 * mm, y, f"Chapter {page}: Station {page}")
        y -= 12 * mm
        pdf.setFont("Helvetica", 10)
        body = [
            f"This chapter covers station {page}. Every station is cleaned at the end of",
            "each shift, and the shift lead signs the log before leaving the kitchen.",
        ]
        if page == 7:
            body = [
                "Fryer oil is changed every 48 hours, or sooner when it smokes or",
                "darkens. Let the oil cool below 50 degrees before draining it.",
            ]
        for line in body:
            pdf.drawString(20 * mm, y, line)
            y -= 5 * mm
        if page == 3:
            y -= 4 * mm
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(20 * mm, y, "Before you start")
            y -= 8 * mm
            pdf.setFont("Helvetica", 10)
            for item in ["Wash your hands", "Put on gloves", "Check the thermometer"]:
                pdf.drawString(22 * mm, y, "\u2022")
                pdf.drawString(27 * mm, y, item)
                y -= 5 * mm
        if page == 5:
            y -= 4 * mm
            pdf.setFont("Helvetica-Bold", 13)
            pdf.drawString(20 * mm, y, "Cleaning schedule")
            y -= 8 * mm
            rows = [
                ("Equipment", "Frequency", "Owner"),
                ("Fryer", "Daily", "Shift lead"),
                ("Grill", "Every shift", "Cook"),
                ("Ice machine", "Weekly", "Manager"),
            ]
            for row in rows:
                pdf.setFont("Helvetica-Bold" if row[0] == "Equipment" else "Helvetica", 10)
                for column, cell in enumerate(row):
                    pdf.drawString((20 + column * 45) * mm, y, cell)
                y -= 6 * mm
            pdf.setFont("Helvetica", 10)
            pdf.drawString(20 * mm, y - 4 * mm, "Log every cleaning in the station binder.")
        pdf.showPage()
    pdf.save()


def columns_pdf() -> None:
    """A full-width title over two columns of prose, read left column first."""
    pdf = canvas.Canvas(str(HERE / "columns.pdf"), pagesize=A4)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(20 * mm, HEIGHT - 30 * mm, "Newsletter")
    left = [
        "The left column starts the story about the",
        "new menu, which launches in every store on",
        "the first of the month with three new items.",
    ]
    right = [
        "The right column carries on with the training",
        "plan: every team member watches the video",
        "and passes the quiz before the launch day.",
    ]
    pdf.setFont("Helvetica", 10)
    for index, (a, b) in enumerate(zip(left, right)):
        y = HEIGHT - 45 * mm - index * 5 * mm
        pdf.drawString(20 * mm, y, a)
        pdf.drawString(110 * mm, y, b)
    pdf.showPage()
    pdf.save()


def page_image(lines: list[str]) -> Image.Image:
    image = Image.new("L", (1240, 1754), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=36)
    for index, line in enumerate(lines):
        draw.text((120, 160 + index * 60), line, fill=0, font=font)
    return image


def scanned_pdf() -> None:
    """A page with text, then a page that is only a picture of one, as a scanner makes it."""
    text = io.BytesIO()
    pdf = canvas.Canvas(text, pagesize=A4)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(20 * mm, HEIGHT - 30 * mm, "This first page has a text layer.")
    pdf.showPage()
    picture = page_image(["Scanned page", "The walk-in freezer stays below -18 C."])
    pdf.drawInlineImage(picture, 0, 0, WIDTH, HEIGHT)
    pdf.showPage()
    pdf.save()
    (HERE / "scanned.pdf").write_bytes(text.getvalue())


def encrypted_pdf() -> None:
    writer = PdfWriter(clone_from=PdfReader(HERE / "columns.pdf"))
    writer.encrypt(user_password="secret", algorithm="AES-128")
    with open(HERE / "encrypted.pdf", "wb") as out:
        writer.write(out)


HANDBOOK = """# Employee Handbook

Welcome to the team. This handbook says what we expect.

## Uniforms

- Wear a clean shirt
- Wear non-slip shoes
    - Black or brown

1. Clock in
2. Wash your hands

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
<w:p><w:r><w:lastRenderedPageBreak/><w:t>Page two starts with the break policy.</w:t></w:r></w:p>
```

## Breaks

| Shift length | Break |
|--------------|-------|
| 6 hours | 30 minutes |
| 8 hours | 45 minutes |

Breaks are taken in the back room.
"""


def office_files() -> None:
    (HERE / "handbook.md").write_text(HANDBOOK)
    subprocess.run(["pandoc", "handbook.md", "-o", "handbook.docx"], cwd=HERE, check=True)
    subprocess.run(["pandoc", "handbook.md", "-s", "-o", "handbook.rtf"], cwd=HERE, check=True)
    odt = HANDBOOK.replace(
        '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'
        "<w:p><w:r><w:lastRenderedPageBreak/><w:t>Page two starts with the break policy.</w:t></w:r></w:p>\n```",
        "```{=opendocument}\n<text:soft-page-break/>\n"
        "<text:p>Page two starts with the break policy.</text:p>\n```",
    )
    (HERE / "handbook-odt.md").write_text(odt)
    subprocess.run(["pandoc", "handbook-odt.md", "-o", "handbook.odt"], cwd=HERE, check=True)
    (HERE / "handbook.md").unlink()
    (HERE / "handbook-odt.md").unlink()

    html = (
        "<html><body><h1>Store Policies</h1><p>Opening hours are 6am to 10pm.</p>"
        "<ul><li>Wash hands</li><li>Wear gloves</li></ul>"
        "<table border='1'><tr><th>Item</th><th>Price</th></tr>"
        "<tr><td>Sandwich</td><td>4.50</td></tr><tr><td>Lemonade</td><td>2.25</td></tr></table>"
        "<p>Refunds are given within 30 days with a receipt.</p></body></html>"
    )
    (HERE / "policies.html").write_text(html)
    subprocess.run(
        ["textutil", "-convert", "doc", "policies.html", "-output", "policies.doc"],
        cwd=HERE,
        check=True,
    )
    (HERE / "policies.html").unlink()

    plain = io.BytesIO((HERE / "handbook.docx").read_bytes())
    office = OOXMLFile(plain)
    with open(HERE / "encrypted.docx", "wb") as out:
        office.encrypt("secret", out)


def menu_xlsx() -> None:
    book = openpyxl.Workbook()
    menu = book.active
    menu.title = "Menu"
    menu.append(["Item", "Price", "Calories"])
    for row in [("Chicken sandwich", 4.5, 440), ("Nuggets (8)", 3.95, 250), ("Lemonade", 2.25, 220)]:
        menu.append(row)
    for cell in menu["B"][1:]:
        cell.number_format = '"$"#,##0.00'
    hours = book.create_sheet("Hours")
    hours.append(["Day", "Opens", "Closes"])
    hours.append(["Monday", "6:00", "22:00"])
    hours.append(["Sunday", "Closed", "Closed"])
    book.create_sheet("Empty")
    book.save(HERE / "menu.xlsx")


if __name__ == "__main__":
    manual_pdf()
    columns_pdf()
    scanned_pdf()
    encrypted_pdf()
    office_files()
    menu_xlsx()
    (HERE / "corrupt.pdf").write_bytes(b"%PDF-1.7\n1 0 obj << /Type /Catalog >>\ntrailer\n%%EOF\n")
