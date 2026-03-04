"""Video frame utilities."""

import io
import logging
from typing import Literal

import av
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Resampling

logger = logging.getLogger(__name__)

TemporalEncoding = Literal["rgb", "heatmap", "grid"]

TEMPORAL_ENCODING_HINTS: dict[str, str] = {
    "rgb": (
        "Video frames are encoded as an RGB temporal composite: the red channel "
        "shows where objects were at the start of the window, green shows the "
        "middle, and blue shows the end. Gray/white areas are static. Color "
        "separation indicates motion — wider separation means faster movement, "
        "and the red→green→blue direction shows the trajectory over time."
    ),
    "heatmap": (
        "The latest video frame is shown with a motion heatmap overlay. "
        "Warm/bright regions indicate where pixel changes occurred during "
        "the preceding time window. The underlying image shows the current "
        "scene; the overlay shows where motion happened."
    ),
    "grid": (
        "Multiple video frames are arranged in a time-ordered grid. "
        "Frames read left-to-right, top-to-bottom from earliest to latest. "
        "Compare object positions across cells to understand motion."
    ),
}


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


def _resize_frame_to_array(
    frame: av.VideoFrame, target_width: int, target_height: int
) -> NDArray[np.uint8]:
    """Resize a frame to target dimensions (maintaining aspect ratio) and return as numpy array."""
    img = frame.to_image()
    src_width, src_height = img.size
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)
    resized = img.resize((new_width, new_height), Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas.paste(resized, (x_offset, y_offset))
    return np.array(canvas, dtype=np.uint8)


def _to_grayscale(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert RGB array (H, W, 3) to grayscale (H, W) using luminance weights."""
    return (
        0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    ).astype(np.uint8)


def _average_arrays(arrays: list[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    """Average a list of same-shape arrays."""
    stacked = np.stack(arrays, axis=0)
    return np.mean(stacked, axis=0).astype(np.uint8)


def frames_to_rgb_temporal_composite(
    frames: list[av.VideoFrame],
    target_width: int = 800,
    target_height: int = 600,
) -> Image.Image:
    """Encode temporal information as color by mapping time to RGB channels.

    Splits frames into three groups (early, middle, late) and assigns each
    group's averaged grayscale to the R, G, and B channels respectively.
    Static regions appear gray/white; moving regions show color separation
    where the R-to-G-to-B shift indicates motion direction over time.

    Requires at least 3 frames. With exactly 3 frames, each frame maps
    directly to one channel with no averaging.

    Args:
        frames: ordered list of video frames (earliest first).
        target_width: output width in pixels.
        target_height: output height in pixels.

    Returns:
        PIL Image with temporal-RGB encoding.
    """
    if len(frames) < 3:
        raise ValueError(f"RGB temporal composite requires >= 3 frames, got {len(frames)}")

    arrays = [
        _resize_frame_to_array(frame, target_width, target_height) for frame in frames
    ]
    grays = [_to_grayscale(arr) for arr in arrays]

    n = len(grays)
    third = n // 3
    remainder = n % 3

    if remainder == 0:
        early, middle, late = grays[:third], grays[third : 2 * third], grays[2 * third :]
    elif remainder == 1:
        early, middle, late = grays[:third], grays[third : 2 * third + 1], grays[2 * third + 1 :]
    else:
        early, middle, late = grays[:third + 1], grays[third + 1 : 2 * third + 1], grays[2 * third + 1 :]

    r_channel = _average_arrays(early)
    g_channel = _average_arrays(middle)
    b_channel = _average_arrays(late)

    composite = np.stack([r_channel, g_channel, b_channel], axis=-1)
    return Image.fromarray(composite)


def frames_to_motion_heatmap(
    frames: list[av.VideoFrame],
    target_width: int = 800,
    target_height: int = 600,
    overlay_alpha: float = 0.4,
) -> Image.Image:
    """Overlay a motion heatmap on the latest frame.

    Computes accumulated frame-to-frame pixel differences across all
    provided frames and renders them as a warm heatmap overlaid on the
    latest frame. Areas with more change glow brighter/warmer.

    Args:
        frames: ordered list of video frames (earliest first), minimum 2.
        target_width: output width in pixels.
        target_height: output height in pixels.
        overlay_alpha: opacity of the heatmap overlay (0.0-1.0).

    Returns:
        PIL Image with the latest frame + motion heatmap overlay.
    """
    if len(frames) < 2:
        raise ValueError(f"Motion heatmap requires >= 2 frames, got {len(frames)}")

    arrays = [
        _resize_frame_to_array(frame, target_width, target_height) for frame in frames
    ]
    grays = [_to_grayscale(arr) for arr in arrays]

    diffs: list[NDArray[np.float64]] = []
    for i in range(1, len(grays)):
        diff = np.abs(grays[i].astype(np.float64) - grays[i - 1].astype(np.float64))
        diffs.append(diff)

    accumulated = np.mean(diffs, axis=0)

    max_val = accumulated.max()
    if max_val > 0:
        normalized = (accumulated / max_val * 255).astype(np.uint8)
    else:
        normalized = accumulated.astype(np.uint8)

    heatmap_rgb = _grayscale_to_heatmap(normalized)

    base = arrays[-1].astype(np.float64)
    heat = heatmap_rgb.astype(np.float64)
    mask = (normalized > 10).astype(np.float64)[:, :, np.newaxis]

    blended = base * (1 - overlay_alpha * mask) + heat * (overlay_alpha * mask)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _grayscale_to_heatmap(gray: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a single-channel intensity map to a warm colormap (black -> red -> yellow -> white)."""
    h, w = gray.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    norm = gray.astype(np.float64) / 255.0
    rgb[:, :, 0] = np.clip(norm * 3.0 * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 1] = np.clip((norm - 0.33) * 3.0 * 255, 0, 255).astype(np.uint8)
    rgb[:, :, 2] = np.clip((norm - 0.66) * 3.0 * 255, 0, 255).astype(np.uint8)
    return rgb


def frames_to_temporal_grid(
    frames: list[av.VideoFrame],
    target_width: int = 800,
    target_height: int = 600,
    cols: int = 2,
) -> Image.Image:
    """Arrange evenly-sampled frames in a grid.

    Selects frames evenly spaced across the input list and arranges them
    in a grid. Each cell is labeled with its relative timestamp position.

    Args:
        frames: ordered list of video frames (earliest first).
        target_width: total output width.
        target_height: total output height.
        cols: number of columns in the grid.

    Returns:
        PIL Image with the frame grid.
    """
    if not frames:
        raise ValueError("At least one frame is required for a grid")

    total_cells = max(cols * ((len(frames) + cols - 1) // cols), cols)
    rows = (total_cells + cols - 1) // cols
    total_cells = rows * cols

    n_to_sample = min(len(frames), total_cells)
    if n_to_sample == 1:
        indices = [0]
    else:
        indices = [
            int(i * (len(frames) - 1) / (n_to_sample - 1)) for i in range(n_to_sample)
        ]

    cell_w = target_width // cols
    cell_h = target_height // rows

    canvas = Image.new("RGB", (target_width, target_height), (0, 0, 0))

    for cell_idx, frame_idx in enumerate(indices):
        frame = frames[frame_idx]
        arr = _resize_frame_to_array(frame, cell_w, cell_h)
        cell_img = Image.fromarray(arr)

        row, col = divmod(cell_idx, cols)
        x = col * cell_w
        y = row * cell_h
        canvas.paste(cell_img, (x, y))

    return canvas


def temporal_composite_to_jpeg_bytes(
    frames: list[av.VideoFrame],
    target_width: int = 800,
    target_height: int = 600,
    mode: str = "rgb",
    quality: int = 85,
) -> bytes:
    """Produce a temporally-encoded JPEG from a list of frames.

    Args:
        frames: ordered list of video frames (earliest first).
        target_width: output width in pixels.
        target_height: output height in pixels.
        mode: encoding mode — "rgb" for RGB temporal composite,
              "heatmap" for motion heatmap overlay on latest frame,
              "grid" for evenly-sampled frame grid.
        quality: JPEG quality (1-100).

    Returns:
        JPEG bytes of the composite image.
    """
    if mode == "rgb":
        img = frames_to_rgb_temporal_composite(frames, target_width, target_height)
    elif mode == "heatmap":
        img = frames_to_motion_heatmap(frames, target_width, target_height)
    elif mode == "grid":
        img = frames_to_temporal_grid(frames, target_width, target_height)
    else:
        raise ValueError(f"Unknown temporal encoding mode: {mode!r}")

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
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
