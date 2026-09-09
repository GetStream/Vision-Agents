"""Video frame utilities."""

import io
import logging

import av
import numpy as np
from PIL import Image
from PIL.Image import Resampling

logger = logging.getLogger(__name__)


def ensure_even_dimensions(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Ensure frame has even dimensions for H.264 yuv420p encoding.

    Crops 1 pixel from right/bottom edge if width/height is odd.
    """
    needs_width_adjust = frame.width % 2 != 0
    needs_height_adjust = frame.height % 2 != 0

    if not needs_width_adjust and not needs_height_adjust:
        return frame

    new_width = frame.width - (1 if needs_width_adjust else 0)
    new_height = frame.height - (1 if needs_height_adjust else 0)

    # Convert to numpy, crop (slice), convert back - faster than reformat which rescales
    arr = frame.to_ndarray(format="rgb24")
    cropped_arr = arr[:new_height, :new_width]
    cropped = av.VideoFrame.from_ndarray(cropped_arr, format="rgb24")
    cropped.pts = frame.pts
    if frame.time_base is not None:
        cropped.time_base = frame.time_base

    return cropped


def frame_to_jpeg_bytes(
    frame: av.VideoFrame, target_width: int, target_height: int, quality: int = 85
) -> bytes:
    """
    Convert a video frame to JPEG bytes with resizing.

    Args:
        frame: an instance of `av.VideoFrame`.
        target_width: target width in pixels.
        target_height: target height in pixels.
        quality: JPEG quality. Default is 85.

    Returns: frame as JPEG bytes.

    """
    # Convert frame to a PIL image
    img = frame.to_image()

    # Calculate scaling to maintain aspect ratio
    src_width, src_height = img.size
    # Calculate scale factor (fit within target dimensions)
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)

    # Resize with aspect ratio maintained
    resized = img.resize((new_width, new_height), Resampling.LANCZOS)

    # Save as JPEG with quality control
    buf = io.BytesIO()
    resized.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def frame_to_png_bytes(frame: av.VideoFrame) -> bytes:
    """
    Convert a video frame to PNG bytes.

    Args:
        frame: Video frame object that can be converted to an image

    Returns:
        PNG bytes of the frame, or empty bytes if conversion fails
    """
    if hasattr(frame, "to_image"):
        img = frame.to_image()
    else:
        arr = frame.to_ndarray(format="rgb24")
        img = Image.fromarray(arr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def resize_frame(frame: av.VideoFrame, width: int, height: int) -> av.VideoFrame:
    """Resize to width x height preserving aspect ratio, padding remainder with black.

    Matched-aspect fast path stays in the source pixel format (yuv420p from WebRTC),
    skipping color conversion. Letterbox path converts to RGB for numpy padding.
    """
    scale = min(width / frame.width, height / frame.height)
    inner_w = max(2, int(frame.width * scale)) & ~1
    inner_h = max(2, int(frame.height * scale)) & ~1

    if inner_w == width and inner_h == height:
        return frame.reformat(width=width, height=height)

    rgb = frame.reformat(width=inner_w, height=inner_h, format="rgb24")
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = (height - inner_h) // 2
    x0 = (width - inner_w) // 2
    out[y0 : y0 + inner_h, x0 : x0 + inner_w] = rgb.to_ndarray()
    return av.VideoFrame.from_ndarray(out, format="rgb24")
