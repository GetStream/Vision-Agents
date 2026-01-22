import asyncio
import logging
from typing import Optional, Callable, Any, cast

import av
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData

logger = logging.getLogger(__name__)


class AudioForwarder:
    """Forwards audio from a MediaStreamTrack to a callback.

    Handles audio frame reading, resampling to 16kHz mono format,
    and forwarding to registered callbacks.

    Audio Format Standardization:
        AudioForwarder ALWAYS resamples audio to 16kHz mono (1 channel) regardless
        of the input format. This standardization is intentional for voice processing:

        - **Sample Rate**: Fixed at 16000 Hz (industry standard for ASR/voice)
        - **Channels**: Fixed at 1 (mono) - suitable for voice applications
        - **Bit Depth**: 16-bit signed integer (int16)

    Why 16kHz Mono?
        This format is hardcoded because AudioForwarder is designed specifically
        for voice processing applications (Automatic Speech Recognition, voice
        assistants, etc.) where:
        - 16kHz provides sufficient quality for human voice (0-8kHz range)
        - Mono reduces bandwidth and processing requirements
        - Most ASR models are trained on 16kHz mono audio
        - Reduces computational cost compared to higher sample rates

    When NOT to Use AudioForwarder:
        - If you need higher sample rates (24kHz, 48kHz) for music/multimedia
        - If you need stereo audio preservation
        - If you want custom format control

        Alternative: Use AudioOutputTrack directly for full format control.

    See Also:
        - AUDIO_DOCUMENTATION.md: Comprehensive audio format guide
        - AudioOutputTrack: For configurable audio format handling
    """

    def __init__(self, track: AudioStreamTrack, callback: Callable[[PcmData], Any]):
        """Initialize the audio forwarder.

        Args:
            track: Audio track to read frames from.
            callback: Async function that receives PcmData (16kHz, mono, int16).
        """
        self.track = track
        self._callback = callback
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start forwarding audio frames to the callback."""
        if self._task is not None:
            logger.warning("AudioForwarder already started")
            return
        self._task = asyncio.create_task(self._reader())

    async def stop(self) -> None:
        """Stop forwarding audio frames."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("AudioForwarder stopped")

    async def _reader(self):
        """Read audio frames from track and forward to callback."""
        while True:
            try:
                received = await asyncio.wait_for(self.track.recv(), timeout=1.0)
                frame = cast(av.AudioFrame, received)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Audio track ended or error: {e}")
                break

            try:
                pcm = PcmData.from_av_frame(frame)

                # CRITICAL FORMAT CONVERSION POINT:
                # Resample ALL incoming audio to 16kHz mono for ASR compatibility.
                # This is hardcoded because AudioForwarder is specifically designed
                # for voice processing applications where 16kHz mono is standard.
                #
                # Format conversion details:
                # - Input: Any sample rate (commonly 48kHz stereo from WebRTC)
                # - Output: Always 16000 Hz, 1 channel (mono), int16
                # - Method: Uses GetStream's built-in resample() with linear interpolation
                # - Channel mixing: Stereo->mono averages the two channels
                #
                # If you need different output formats, use AudioOutputTrack directly.
                pcm = pcm.resample(target_sample_rate=16000, target_channels=1)
                await self._callback(pcm)
            except Exception as e:
                logger.exception(f"Failed to process audio frame: {e}")
