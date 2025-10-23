import asyncio
import os
import shutil
import tempfile
import time
from typing import Optional

from vision_agents.core.tts import TTS
from vision_agents.core.tts.testing import TTSSession
from vision_agents.core.edge.types import PcmData


async def manual_tts_to_wav(
    tts: TTS,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    text: str = "This is a manual TTS playback test.",
    outfile_path: Optional[str] = None,
    timeout_s: float = 20.0,
    play_env: str = "FFPLAY",
) -> str:
    """Generate TTS audio to a WAV file and optionally play with ffplay.

    - Creates the TTS instance via `tts_factory()`.
    - Sets desired output format via `set_output_format(sample_rate, channels)`.
    - Sends `text` and captures TTSAudioEvent chunks.
    - Writes a WAV (s16) file and returns the path.
    - If env `play_env` is set to "1" and `ffplay` exists, it plays the file.

    Args:
        tts: the TTS instance.
        sample_rate: desired sample rate to write.
        channels: desired channels to write.
        text: text to synthesize.
        outfile_path: optional absolute path for the WAV file; if None, temp path.
        timeout_s: timeout for first audio to arrive.
        play_env: env var name controlling playback (default: FFPLAY).

    Returns:
        Path to written WAV file.
    """

    tts.set_output_format(sample_rate=sample_rate, channels=channels)
    session = TTSSession(tts)
    await tts.send(text)
    result = await session.wait_for_result(timeout=timeout_s)
    if result.errors:
        raise RuntimeError(f"TTS errors: {result.errors}")

    # Write WAV file (16kHz mono, s16)
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(
            tmpdir, f"tts_manual_test_{tts.__class__.__name__}_{timestamp}.wav"
        )

    pcm_bytes = b"".join(result.speeches)
    pcm = PcmData.from_bytes(
        pcm_bytes, sample_rate=sample_rate, channels=channels, format="s16"
    )
    with open(outfile_path, "wb") as f:
        f.write(pcm.to_wav_bytes())

    # Optional playback
    if os.environ.get(play_env) == "1" and shutil.which("ffplay"):
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
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            proc.kill()

    return outfile_path
