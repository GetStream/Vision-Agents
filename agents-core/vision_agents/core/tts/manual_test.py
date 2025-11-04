from typing import Optional

import aiofiles
from getstream.video.rtc import PcmData
from vision_agents.core.tts import TTS
from vision_agents.core.tts.testing import TTSSession
import asyncio
import os
import shutil
import tempfile
import time
import logging

logger = logging.getLogger(__name__)


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


async def manual_tts_to_wav(
    tts: TTS,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    text: str = "This is a manual TTS playback test.",
    outfile_path: Optional[str] = None,
    timeout_s: float = 20.0,
) -> str:
    """Generate TTS audio to a WAV file and optionally play with ffplay.

    - Receives a TTS instance.
    - Configures desired output format via `set_output_format(sample_rate, channels)`.
    - Sends `text` and captures TTSAudioEvent chunks.
    - Writes a WAV (s16) file and returns the path.
    - If `ffplay` exists, it plays the file.

    Args:
        tts: the TTS instance.
        sample_rate: desired sample rate to write.
        channels: desired channels to write.
        text: text to synthesize.
        outfile_path: optional absolute path for the WAV file; if None, temp path.
        timeout_s: timeout for first audio to arrive.

    Returns:
        Path to written WAV file.
    """

    tts.set_output_format(sample_rate=sample_rate, channels=channels)
    session = TTSSession(tts)
    await tts.send(text)
    result = await session.wait_for_result(timeout=timeout_s)
    if result.errors:
        raise RuntimeError(f"TTS errors: {result.errors}")

    if len(result.speeches) == 0:
        return ""

    pcm = result.speeches[0]
    [pcm.append(p) for p in result.speeches[1:]]

    # Generate a descriptive filename if not provided
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(
            tmpdir, f"tts_manual_test_{tts.__class__.__name__}_{timestamp}.wav"
        )

    # Use utility function to write WAV and optionally play
    return await play_pcm_with_ffplay(pcm, outfile_path=outfile_path, timeout_s=30.0)
