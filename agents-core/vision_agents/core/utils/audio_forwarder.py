import asyncio
import logging
from typing import Optional, Callable, Any, cast

import av
import numpy as np
from av.frame import Frame
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.recording import AudioFrame
from getstream.video.rtc.track_util import PcmData, AudioFormat

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
        logger.info("AudioForwarder started")

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
                # Convert Frame to numpy array
                samples = frame.to_ndarray()
                
                # Convert stereo to mono if needed
                if samples.ndim == 2 and samples.shape[0] > 1:
                    samples = samples.mean(axis=0)
                
                # Ensure int16 format
                if samples.dtype != np.int16:
                    samples = samples.astype(np.int16)
                
                # Create PcmData from the frame
                pcm = PcmData(samples=samples, sample_rate=frame.sample_rate, format=AudioFormat.S16)
                
                # Resample to 16kHz mono
                pcm = pcm.resample(16000, 1)
                
                # Forward to callback
                await self._callback(pcm)
            except Exception as e:
                logger.debug(f"Failed to process audio frame: {e}")
