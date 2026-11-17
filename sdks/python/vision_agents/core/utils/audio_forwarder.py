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
                # Resample to 16kHz mono as documented in the class docstring
                pcm = pcm.resample(target_sample_rate=16000, target_channels=1)
                await self._callback(pcm)
            except Exception as e:
                logger.exception(f"Failed to process audio frame: {e}")
