"""Tkinter-based video display for LocalEdge.

Shows the agent's outbound video track in a tkinter window.
Gracefully handles environments where tkinter is not available.
"""

import asyncio
import contextlib
import logging
import signal
import threading
import warnings
from typing import cast

import av
from aiortc import MediaStreamError, MediaStreamTrack

logger = logging.getLogger(__name__)

try:
    import tkinter

    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False


def _fit_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Compute the largest size that fits dst while preserving src aspect ratio."""
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(2, int(src_w * scale)) & ~1, max(2, int(src_h * scale)) & ~1


def _frame_to_ppm(frame: av.VideoFrame, width: int, height: int) -> bytes:
    """Convert an av.VideoFrame to PPM bytes, scaled to fit width x height."""
    fit_w, fit_h = _fit_size(frame.width, frame.height, width, height)
    rgb = frame.reformat(width=fit_w, height=fit_h, format="rgb24")
    pixels = rgb.to_ndarray()
    header = f"P6 {fit_w} {fit_h} 255 ".encode()
    return header + pixels.tobytes()


class VideoDisplay:
    """Displays video frames from a MediaStreamTrack in a tkinter window.

    Tkinter events are pumped from the asyncio event loop (main thread)
    to satisfy macOS Cocoa requirements. A thread-safe queue bridges the
    async frame receiver and the display update loop.
    """

    def __init__(
        self,
        title: str = "Agent Video Output",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if width <= 0:
            raise ValueError("width must be > 0")
        if height <= 0:
            raise ValueError("height must be > 0")
        self._title = title
        self._width = width
        self._height = height
        self._frame_interval = 1.0 / fps
        self._running = False
        self._latest_frame: av.VideoFrame | None = None
        self._frame_lock = threading.Lock()
        self._recv_task: asyncio.Task[None] | None = None
        self._tk_task: asyncio.Task[None] | None = None
        self._root: tkinter.Tk | None = None

    async def start(self, video_track: MediaStreamTrack) -> None:
        """Start displaying frames from the given video track.

        If tkinter is not available, emits an ImportWarning and returns
        without starting.
        """
        if not _TKINTER_AVAILABLE:
            warnings.warn(
                "tkinter is not available. Install python3-tk or equivalent "
                "for your platform to use the video display.",
                ImportWarning,
            )
            return

        self._running = True
        self._recv_task = asyncio.create_task(self._recv_loop(video_track))
        self._tk_task = asyncio.create_task(self._tk_loop())

    async def stop(self) -> None:
        """Stop the display and clean up tasks."""
        self._running = False

        for task in (self._recv_task, self._tk_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._recv_task = None
        self._tk_task = None

    async def _recv_loop(self, track: MediaStreamTrack) -> None:
        """Consume frames from the video track and store the latest."""
        try:
            while self._running:
                frame = cast(av.VideoFrame, await track.recv())
                with self._frame_lock:
                    self._latest_frame = frame
        except asyncio.CancelledError:
            raise
        except MediaStreamError:
            logger.debug("Video track ended")
        except RuntimeError:
            logger.debug("Video track stopped")

    async def _tk_loop(self) -> None:
        """Pump Tkinter events from the asyncio event loop (main thread)."""
        root: tkinter.Tk | None = None
        prev_sigint = signal.getsignal(signal.SIGINT)
        try:
            root = tkinter.Tk()
            # Tk() overrides SIGINT — restore the previous handler (typically
            # asyncio's) so the first Ctrl+C gracefully cancels the main task
            # instead of raising KeyboardInterrupt inside this task.
            signal.signal(signal.SIGINT, prev_sigint)
            root.title(self._title)
            root.geometry(f"{self._width}x{self._height}")
            root.protocol("WM_DELETE_WINDOW", self._on_window_close)
            self._root = root

            gray = bytes([128] * (self._width * self._height * 3))
            header = f"P6 {self._width} {self._height} 255 ".encode()
            self._photo = tkinter.PhotoImage(data=header + gray)

            self._label = tkinter.Label(root, image=self._photo)
            self._label.pack(fill="both", expand=True)

            while self._running:
                with self._frame_lock:
                    frame = self._latest_frame
                    self._latest_frame = None

                if frame is not None:
                    ppm = await asyncio.to_thread(
                        _frame_to_ppm,
                        frame,
                        self._width,
                        self._height,
                    )
                    self._photo = tkinter.PhotoImage(data=ppm)
                    self._label.configure(image=self._photo)

                try:
                    root.update()
                except (tkinter.TclError, KeyboardInterrupt):
                    break

                await asyncio.sleep(self._frame_interval)
        except asyncio.CancelledError:
            raise
        finally:
            if root is not None:
                with contextlib.suppress(tkinter.TclError, KeyboardInterrupt):
                    root.destroy()
            self._root = None

    def _on_window_close(self) -> None:
        """Handle the user closing the tkinter window."""
        self._running = False
        if self._root is not None:
            try:
                self._root.destroy()
            except tkinter.TclError:
                pass
            self._root = None
