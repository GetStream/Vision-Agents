"""Standalone Gemini VLM demo — no GetStream SFU required.

Feeds webcam frames (or synthetic frames) directly to the VLM
and prints the model's description.

Usage:
    uv run plugins/gemini/example/gemini_vlm_standalone_example.py
"""

import asyncio
import logging
import os

import av
import numpy as np
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.gemini import VLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-lite-preview"


def capture_webcam_frames(count: int = 3) -> list[av.VideoFrame]:
    """Capture frames from the default webcam via PyAV."""
    frames: list[av.VideoFrame] = []
    try:
        container = av.open("/dev/video0", format="v4l2")
    except (av.error.FileNotFoundError, av.error.InvalidDataError):
        try:
            container = av.open("0", format="avfoundation")
        except (av.error.FileNotFoundError, av.error.InvalidDataError):
            logger.warning("No webcam found, falling back to synthetic frames")
            return []

    for i, frame in enumerate(container.decode(video=0)):
        frames.append(frame)
        if i >= count - 1:
            break
    container.close()
    return frames


def make_synthetic_frames(count: int = 3) -> list[av.VideoFrame]:
    """Generate colored test frames as a fallback."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    frames: list[av.VideoFrame] = []
    for i in range(count):
        arr = np.zeros((240, 320, 3), dtype=np.uint8)
        arr[:, :] = colors[i % len(colors)]
        frames.append(av.VideoFrame.from_ndarray(arr, format="rgb24"))
    return frames


async def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment")
        return

    vlm = VLM(model=MODEL, api_key=api_key)
    vlm.set_conversation(InMemoryConversation("Describe what you see concisely.", []))

    frames = capture_webcam_frames() or make_synthetic_frames()
    for frame in frames:
        vlm._frame_buffer.append(frame)

    logger.info(f"Loaded {len(frames)} frames, sending to {MODEL}...")

    try:
        response = await vlm.simple_response("What do you see in these frames?")
        print(f"\n--- {MODEL} response ---")
        print(response.text)
        print("---")
    finally:
        await vlm.close()


if __name__ == "__main__":
    asyncio.run(main())
