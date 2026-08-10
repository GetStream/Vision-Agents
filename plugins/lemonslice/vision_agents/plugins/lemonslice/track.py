import asyncio
import fractions

import aiortc
import av
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, AudioFormatType, PcmData

__all__ = ["AvatarInputTrack"]


class AvatarInputTrack(AudioStreamTrack):
    """
    An input audio track for the LemonSlice avatar.

    Key differences from the base AudioStreamTrack:
    - Tracks the last produced "pts" and exposes it as public API.
    - Returns all available data on "recv()" instead of real-time pacing
      to reduce avatar latency.
    - Waits for the next write instead of synthesizing silence: an avatar has
      nothing to animate during a gap, and the PTS timeline stays contiguous
      across it.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        format: AudioFormatType = AudioFormat.S16,
        audio_buffer_size_ms: int = 30000,
    ):
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            format=format,
            audio_buffer_size_ms=audio_buffer_size_ms,
        )
        # PTS the next emitted frame will carry, in samples.
        self._next_pts = 0
        self._data_available = asyncio.Event()

    async def pts(self) -> int:
        async with self._frame_lock:
            # next PTS to emit + samples still queued = wire PTS once the buffer drains
            return self._next_pts + self._buffered_samples

    async def write(self, pcm: PcmData, final: bool = False) -> None:
        await super().write(pcm, final)
        self._data_available.set()

    async def recv(self) -> av.AudioFrame:
        """Drain buffered audio without pacing, waiting for more instead of emitting silence."""
        if self.readyState != "live":
            raise aiortc.mediastreams.MediaStreamError

        while True:
            async with self._frame_lock:
                if not self._frame_buffer:
                    # Starved: emit the resampler's partial tail instead of waiting for a full frame.
                    for tail in self._resampler.flush():
                        self._frame_buffer.append(tail)
                        self._buffered_samples += tail.samples
                if self._frame_buffer:
                    frame = self._frame_buffer.popleft()
                    self._buffered_samples -= frame.samples
                    break
                self._data_available.clear()
            await self._data_available.wait()

        frame.pts = self._next_pts
        # Advance by the samples actually emitted; a short tail must not consume a
        # full frame slot, or the encoder's resampler pads the gap with silence.
        self._next_pts += frame.samples
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame
