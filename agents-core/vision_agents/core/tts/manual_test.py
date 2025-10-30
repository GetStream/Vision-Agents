import os
import tempfile
import time

from typing import Optional

from vision_agents.core.tts import TTS
from vision_agents.core.tts.testing import TTSSession
from getstream.video.rtc.track_util import PcmData, AudioFormat
from vision_agents.core.edge.types import play_pcm_with_ffplay


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

    # Convert captured audio to PcmData
    pcm_bytes = b"".join(result.speeches)
    pcm = PcmData.from_bytes(
        pcm_bytes, sample_rate=sample_rate, channels=channels, format=AudioFormat.S16
    )

    # Generate a descriptive filename if not provided
    if outfile_path is None:
        tmpdir = tempfile.gettempdir()
        timestamp = int(time.time())
        outfile_path = os.path.join(
            tmpdir, f"tts_manual_test_{tts.__class__.__name__}_{timestamp}.wav"
        )

    # Use utility function to write WAV and optionally play
    return await play_pcm_with_ffplay(pcm, outfile_path=outfile_path, timeout_s=30.0)
