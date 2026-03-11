"""Configurable grid overlay and cell-to-coordinate conversion."""

import logging
import re

import av
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageFont import FreeTypeFont, ImageFont as PILImageFont

logger = logging.getLogger(__name__)

_LINE_COLOR = (255, 255, 0, 100)
_LABEL_COLOR = (255, 255, 0, 200)

_POSITION_OFFSETS: dict[str, tuple[float, float]] = {
    "top-left": (0.2, 0.2),
    "top": (0.5, 0.2),
    "top-right": (0.8, 0.2),
    "left": (0.2, 0.5),
    "center": (0.5, 0.5),
    "right": (0.8, 0.5),
    "bottom-left": (0.2, 0.8),
    "bottom": (0.5, 0.8),
    "bottom-right": (0.8, 0.8),
}

VIRTUAL_SIZE = 1000


class Grid:
    """A configurable grid for mapping cell references to screen coordinates.

    Args:
        cols: Number of columns (1-26). Default 15.
        rows: Number of rows (1-99). Default 15.
    """

    def __init__(self, cols: int = 15, rows: int = 15):
        if cols < 1 or cols > 26:
            raise ValueError(f"cols must be 1-26, got {cols}")
        if rows < 1 or rows > 99:
            raise ValueError(f"rows must be 1-99, got {rows}")

        self.cols = cols
        self.rows = rows
        self.col_labels = [chr(ord("A") + i) for i in range(cols)]
        self.row_labels = list(range(1, rows + 1))
        self._cell_w = VIRTUAL_SIZE // cols
        self._cell_h = VIRTUAL_SIZE // rows

        last_col = self.col_labels[-1]
        self._cell_pattern = re.compile(
            rf"^([A-{last_col}a-{last_col.lower()}])(\d{{1,2}})$"
        )

    @property
    def label(self) -> str:
        """Short description, e.g. 'A-O / 1-15'."""
        return f"{self.col_labels[0]}-{self.col_labels[-1]} / 1-{self.rows}"

    def cell_to_virtual(self, cell: str, position: str = "center") -> tuple[int, int]:
        """Convert a cell reference like 'C2' to virtual (x, y).

        Args:
            cell: Grid cell, e.g. "C2".
            position: Sub-cell target. One of: top-left, top, top-right, left,
                center, right, bottom-left, bottom, bottom-right.

        Raises:
            ValueError: If the cell reference or position is invalid.
        """
        m = self._cell_pattern.match(cell.strip())
        if not m:
            raise ValueError(f"Invalid cell reference: {cell!r}. Use format like 'C2'.")
        col_letter = m.group(1).upper()
        row_num = int(m.group(2))

        if col_letter not in self.col_labels:
            raise ValueError(
                f"Column must be {self.col_labels[0]}-{self.col_labels[-1]}, "
                f"got {col_letter!r}"
            )
        if row_num < 1 or row_num > self.rows:
            raise ValueError(f"Row must be 1-{self.rows}, got {row_num}")

        offsets = _POSITION_OFFSETS.get(position.lower().strip())
        if offsets is None:
            raise ValueError(
                f"Invalid position: {position!r}. "
                f"Choose from: {', '.join(_POSITION_OFFSETS)}"
            )
        ox, oy = offsets

        col_idx = self.col_labels.index(col_letter)
        row_idx = row_num - 1
        vx = int(col_idx * self._cell_w + self._cell_w * ox)
        vy = int(row_idx * self._cell_h + self._cell_h * oy)
        return vx, vy

    def draw_overlay(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Draw a labeled grid overlay on *frame*."""
        img = frame.to_image().convert("RGBA")
        w, h = img.size

        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font: FreeTypeFont | PILImageFont
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Menlo.ttc",
                size=max(8, h // (self.rows * 4)),
            )
        except OSError:
            font = ImageFont.load_default()

        col_step = w / self.cols
        row_step = h / self.rows

        for i in range(1, self.cols):
            x = int(i * col_step)
            draw.line([(x, 0), (x, h)], fill=_LINE_COLOR, width=1)

        for i in range(1, self.rows):
            y = int(i * row_step)
            draw.line([(0, y), (w, y)], fill=_LINE_COLOR, width=1)

        for ci, col in enumerate(self.col_labels):
            for ri, row in enumerate(self.row_labels):
                label = f"{col}{row}"
                lx = int(ci * col_step) + 2
                ly = int(ri * row_step) + 1
                draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0, 160), font=font)
                draw.text((lx, ly), label, fill=_LABEL_COLOR, font=font)

        composited = Image.alpha_composite(img, overlay).convert("RGB")
        return av.VideoFrame.from_image(composited)
