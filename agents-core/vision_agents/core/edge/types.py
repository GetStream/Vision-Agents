from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)
import logging

from getstream.video.rtc.track_util import PcmData
from pyee.asyncio import AsyncIOEventEmitter
import asyncio
import os
import shutil
import tempfile
import time

logger = logging.getLogger(__name__)


@dataclass
class User:
    id: Optional[str] = ""
    name: Optional[str] = ""
    image: Optional[str] = ""


@dataclass
class Participant:
    original: Any
    user_id: str


class Connection(AsyncIOEventEmitter):
    """
    To standardize we need to have a method to close
    and a way to receive a callback when the call is ended
    In the future we might want to forward more events
    """

    async def close(self):
        pass


@runtime_checkable
class OutputAudioTrack(Protocol):
    """
    A protocol describing an output audio track, the actual implementation depends on the edge transported used
    eg. getstream.video.rtc.audio_track.AudioStreamTrack
    """

    async def write(self, data: bytes) -> None: ...

    def stop(self) -> None: ...


async def play_pcm_with_ffplay(
    pcm: PcmData,
    outfile_path: Optional[str] = None,
    timeout_s: float = 30.0,
) -> str:
    """Write PcmData to a WAV file and optionally play it with ffplay.

    This is a utility function for testing and debugging audio output.

    Args:
        pcm: PcmData object to play
        outfile_path: Optional path for the WAV file. If None, creates a temp file.
        timeout_s: Timeout in seconds for ffplay playback (default: 30.0)

    Returns:
        Path to the written WAV file

    Example:
        pcm = PcmData.from_bytes(audio_bytes, sample_rate=48000, channels=2)
        wav_path = await play_pcm_with_ffplay(pcm)
    """

    # Generate output path if not provided
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(tmpdir, f"pcm_playback_{timestamp}.wav")

    # Write WAV file
    with open(outfile_path, "wb") as f:
        f.write(pcm.to_wav_bytes())

    logger.info(f"Wrote WAV file: {outfile_path}")

    # Optional playback with ffplay
    if shutil.which("ffplay"):
        logger.info("Playing audio with ffplay...")
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-autoexit",
            "-nodisp",
            "-hide_banner",
            "-loglevel",
            "error",
            outfile_path,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(f"ffplay timed out after {timeout_s}s, killing process")
            proc.kill()
    else:
        logger.warning("ffplay not found in PATH, skipping playback")

    return outfile_path
