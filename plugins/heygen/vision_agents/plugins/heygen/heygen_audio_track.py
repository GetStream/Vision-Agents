"""Custom audio track for sending audio to HeyGen for lip-sync."""

import asyncio
import logging
from typing import Optional
from fractions import Fraction

import av
import numpy as np
from aiortc import AudioStreamTrack

logger = logging.getLogger(__name__)


class HeyGenAudioTrack(AudioStreamTrack):
    """Audio track that accepts PCM data and produces frames for WebRTC.
    
    This track receives audio data from the Realtime LLM and produces
    audio frames that can be sent to HeyGen via WebRTC for lip-sync.
    """
    
    kind = "audio"
    
    def __init__(self, sample_rate: int = 24000):
        """Initialize the audio track.
        
        Args:
            sample_rate: Sample rate for audio frames (default: 24000 for Gemini).
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._ts = 0
        self._latest_chunk: Optional[bytes] = None
        self._silence_cache: dict[int, np.ndarray] = {}
        logger.info(f"🎤 HeyGenAudioTrack initialized at {sample_rate}Hz")
    
    def write_audio(self, pcm_data: bytes) -> None:
        """Write PCM audio data to be sent to HeyGen.
        
        Args:
            pcm_data: Raw PCM16 audio data from the LLM.
        """
        if not pcm_data:
            return
        self._latest_chunk = bytes(pcm_data)
        logger.debug(f"✍️ Audio data written: {len(pcm_data)} bytes")
    
    async def recv(self) -> av.AudioFrame:
        """Receive the next audio frame for WebRTC transmission.
        
        Returns:
            Audio frame to send to HeyGen.
        """
        # Pace at 20ms per frame (50 fps)
        await asyncio.sleep(0.02)
        
        sr = self._sample_rate
        samples_per_frame = int(0.02 * sr)  # 20ms worth of samples
        
        chunk = self._latest_chunk
        if chunk:
            logger.debug(f"🎙️ recv() producing frame with audio data ({len(chunk)} bytes)")
        if chunk:
            # Consume and clear the latest pushed chunk
            self._latest_chunk = None
            arr = np.frombuffer(chunk, dtype=np.int16)
            
            # Ensure mono channel
            if arr.ndim == 1:
                samples = arr.reshape(1, -1)
            else:
                samples = arr[:1, :]
            
            # Pad or truncate to exactly one 20ms frame
            needed = samples_per_frame
            have = samples.shape[1]
            if have < needed:
                pad = np.zeros((1, needed - have), dtype=np.int16)
                samples = np.concatenate([samples, pad], axis=1)
            elif have > needed:
                samples = samples[:, :needed]
        else:
            # Generate silence when no audio data is available
            cached = self._silence_cache.get(sr)
            if cached is None:
                cached = np.zeros((1, samples_per_frame), dtype=np.int16)
                self._silence_cache[sr] = cached
            samples = cached
        
        # Create audio frame
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = sr
        frame.pts = self._ts
        frame.time_base = Fraction(1, sr)
        self._ts += samples.shape[1]
        
        return frame

