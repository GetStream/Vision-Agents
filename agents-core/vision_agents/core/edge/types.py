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
import aiofiles
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

    async def write(self, data: PcmData) -> None: ...

    def stop(self) -> None: ...


async def play_pcm_with_ffplay(
    pcm: PcmData,
    outfile_path: Optional[str] = None,
    timeout_s: float = 30.0,
) -> str:
    """Write PcmData to a WAV file and optionally play it with ffplay.

    This is a utility function for testing and debugging audio output.
    Audio playback only happens if PLAY_AUDIO environment variable is set to "true".

    Args:
        pcm: PcmData object to play
        outfile_path: Optional path for the WAV file. If None, creates a temp file.
        timeout_s: Timeout in seconds for ffplay playback (default: 30.0)

    Returns:
        Path to the written WAV file

    Example:
        pcm = PcmData.from_bytes(audio_bytes, sample_rate=48000, channels=2)
        wav_path = await play_pcm_with_ffplay(pcm)
        
    Note:
        Set PLAY_AUDIO=true environment variable to enable audio playback during tests.
    """

    # Generate output path if not provided
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(tmpdir, f"pcm_playback_{timestamp}.wav")

    # Write WAV file asynchronously
    async with aiofiles.open(outfile_path, "wb") as f:
        await f.write(pcm.to_wav_bytes())

    logger.info(f"Wrote WAV file: {outfile_path}")

    # Optional playback with ffplay - only if PLAY_AUDIO environment variable is set
    play_audio = os.environ.get("PLAY_AUDIO", "").lower() in ("true", "1", "yes")
    
    if play_audio:
        # Check in thread pool to avoid blocking
        has_ffplay = await asyncio.to_thread(shutil.which, "ffplay")
        if has_ffplay:
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
    else:
        logger.debug("Skipping audio playback (set PLAY_AUDIO=true to enable)")

    return outfile_path
