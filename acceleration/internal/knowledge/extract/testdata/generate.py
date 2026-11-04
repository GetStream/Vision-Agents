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
import openpyxl.drawing.image
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
        pdf.drawString(
            20 * mm, HEIGHT - 12 * mm, "Equipment Manual - Kitchen Operations"
        )
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
                pdf.setFont(
                    "Helvetica-Bold" if row[0] == "Equipment" else "Helvetica", 10
                )
                for column, cell in enumerate(row):
                    pdf.drawString((20 + column * 45) * mm, y, cell)
                y -= 6 * mm
            pdf.setFont("Helvetica", 10)
            pdf.drawString(
                20 * mm, y - 4 * mm, "Log every cleaning in the station binder."
            )
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


def chart_image(width: int, height: int) -> Image.Image:
    """A bar chart as a picture, the way one pasted into a document is."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(12, height // 16))
    draw.text((width // 20, height // 30), "Fryer oil changes", fill="black", font=font)
    for index, value in enumerate([0.4, 0.7, 0.55, 0.9]):
        left = width // 8 + index * width // 5
        top = int(height * (0.9 - 0.7 * value))
        draw.rectangle(
            [left, top, left + width // 8, int(height * 0.9)], fill="steelblue"
        )
    return image


def figures_pdf() -> None:
    """A page each of what does and does not make a page figured."""
    pdf = canvas.Canvas(str(HERE / "figures.pdf"), pagesize=A4)

    def heading(text: str) -> None:
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(20 * mm, HEIGHT - 30 * mm, text)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(
            20 * mm,
            HEIGHT - 40 * mm,
            "Every station is cleaned at the end of each shift.",
        )

    heading("A bar chart")
    for index, value in enumerate([60, 110, 85, 140]):
        pdf.setFillColorRGB(0.2, 0.4, 0.7)
        pdf.rect((30 + index * 35) * mm, 60 * mm, 20 * mm, value * mm, stroke=0, fill=1)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString((32 + index * 35) * mm, 54 * mm, f"Week {index + 1}")
    pdf.showPage()

    heading("A table")
    pdf.setFillColorRGB(0.85, 0.85, 0.85)
    pdf.rect(20 * mm, 200 * mm, 150 * mm, 10 * mm, stroke=0, fill=1)
    pdf.setFillColorRGB(0, 0, 0)
    for row in range(8):
        y = 200 * mm - row * 10 * mm
        for column, text in enumerate(["Fryer", "Daily", "Shift lead"]):
            pdf.drawString((22 + column * 50) * mm, y + 3 * mm, text)
    for row in range(9):
        pdf.line(20 * mm, (210 - row * 10) * mm, 170 * mm, (210 - row * 10) * mm)
    for column in range(4):
        x = (20 + column * 50) * mm
        pdf.line(x, 130 * mm, x, 210 * mm)
    pdf.showPage()

    heading("A pie chart")
    pdf.setFillColorRGB(0.8, 0.3, 0.2)
    pdf.wedge(40 * mm, 60 * mm, 160 * mm, 180 * mm, 0, 120, stroke=0, fill=1)
    pdf.setFillColorRGB(0.2, 0.6, 0.3)
    pdf.wedge(40 * mm, 60 * mm, 160 * mm, 180 * mm, 120, 240, stroke=0, fill=1)
    pdf.showPage()

    heading("A logo")
    pdf.drawInlineImage(chart_image(200, 200), 170 * mm, 270 * mm, 20 * mm, 20 * mm)
    pdf.showPage()

    heading("A photo")
    pdf.drawInlineImage(chart_image(800, 600), 20 * mm, 60 * mm, 160 * mm, 120 * mm)
    pdf.showPage()

    pdf.setFillColorRGB(0.95, 0.93, 0.85)
    pdf.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    pdf.setFillColorRGB(0, 0, 0)
    heading("A tinted page")
    pdf.showPage()

    pdf.beginForm("chart")
    for index, value in enumerate([60, 110, 85, 140]):
        pdf.setFillColorRGB(0.2, 0.4, 0.7)
        pdf.rect((30 + index * 35) * mm, 60 * mm, 20 * mm, value * mm, stroke=0, fill=1)
    pdf.endForm()
    heading("A chart drawn as a form")
    pdf.doForm("chart")
    pdf.showPage()
    pdf.save()


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
    subprocess.run(
        ["pandoc", "handbook.md", "-o", "handbook.docx"], cwd=HERE, check=True
    )
    subprocess.run(
        ["pandoc", "handbook.md", "-s", "-o", "handbook.rtf"], cwd=HERE, check=True
    )
    odt = HANDBOOK.replace(
        '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n'
        "<w:p><w:r><w:lastRenderedPageBreak/><w:t>Page two starts with the break policy.</w:t></w:r></w:p>\n```",
        "```{=opendocument}\n<text:soft-page-break/>\n"
        "<text:p>Page two starts with the break policy.</text:p>\n```",
    )
    (HERE / "handbook-odt.md").write_text(odt)
    subprocess.run(
        ["pandoc", "handbook-odt.md", "-o", "handbook.odt"], cwd=HERE, check=True
    )
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


FIGURES = """# Fryer guide

The fryer is cleaned every night.

![](chart.png)

An icon marks every safety step: ![](icon.png)

| Week | Changes |
|------|---------|
| 1 | ![](map.gif) |

The same chart again, for the night shift:

![](chart.png)
"""


def office_figures() -> None:
    """Word and OpenDocument files with a chart, an icon, a picture in a table and a
    chart placed twice, and a workbook with a chart pasted as a picture."""
    chart_image(800, 600).save(HERE / "chart.png")
    Image.new("RGB", (40, 40), "red").save(HERE / "icon.png")
    chart_image(300, 200).convert("P").save(HERE / "map.gif")
    (HERE / "figures.md").write_text(FIGURES)
    for out in ["figures.docx", "figures.odt"]:
        subprocess.run(["pandoc", "figures.md", "-o", out], cwd=HERE, check=True)

    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Oil"
    sheet.append(["Week", "Changes"])
    sheet.append([1, 4])
    sheet.add_image(openpyxl.drawing.image.Image(str(HERE / "chart.png")), "D2")
    book.save(HERE / "figures.xlsx")

    for name in ["chart.png", "icon.png", "map.gif", "figures.md"]:
        (HERE / name).unlink()


def menu_xlsx() -> None:
    book = openpyxl.Workbook()
    menu = book.active
    menu.title = "Menu"
    menu.append(["Item", "Price", "Calories"])
    for row in [
        ("Chicken sandwich", 4.5, 440),
        ("Nuggets (8)", 3.95, 250),
        ("Lemonade", 2.25, 220),
    ]:
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
    figures_pdf()
    encrypted_pdf()
    office_files()
    office_figures()
    menu_xlsx()
    (HERE / "corrupt.pdf").write_bytes(
        b"%PDF-1.7\n1 0 obj << /Type /Catalog >>\ntrailer\n%%EOF\n"
    )
