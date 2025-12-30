"""Video frame utilities."""

import io
import logging

import av
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


def resize_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
    """
    Resizes a video frame to target dimensions while maintaining the aspect ratio. The method centers the resized
    image on a black background if the target dimensions do not match the original aspect ratio.

    Parameters:
        frame (av.VideoFrame): The input video frame to be resized.

    Returns:
        av.VideoFrame: The output video frame after resizing, maintaining the original aspect ratio.

    Raises:
        None
    """
    img = frame.to_image()

    # Calculate scaling to maintain aspect ratio
    src_width, src_height = img.size
    target_width, target_height = self.width, self.height

    # Calculate scale factor (fit within target dimensions)
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)

    # Resize with aspect ratio maintained
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create black background at target resolution
    result = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    # Paste resized image centered
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    result.paste(resized, (x_offset, y_offset))

    return av.VideoFrame.from_image(result)
