import asyncio
import fractions

import aiortc
import av
from getstream.video.rtc import AudioStreamTrack

__all__ = ["AvatarInputTrack"]


class AvatarInputTrack(AudioStreamTrack):
    """
    An input audio track for the LemonSlice avatar.

    Key differences from the base AudioStreamTrack:
    - Tracks the last produced "pts" and exposes it as public API.
    - Returns all available data on "recv()" instead of real-time pacing
      to reduce avatar latency.
    """

    # PTS of the last emitted frame; None until the first recv().
    _timestamp: int | None = None

    async def pts(self) -> int:
        async with self._frame_lock:
            # last emitted PTS + samples still queued = wire PTS of the last buffered sample
            ts = self._timestamp or 0
            return ts + self._buffered_samples

    async def recv(self) -> av.AudioFrame:
        """Drain buffered audio without pacing; pace only when emitting silence."""
        if self.readyState != "live":
            raise aiortc.mediastreams.MediaStreamError

        if self._timestamp is None:
            self._timestamp = 0
        else:
            self._timestamp += self._samples_per_frame

        async with self._frame_lock:
            if not self._frame_buffer:
                # Starved: emit the resampler's partial tail instead of waiting for a full frame.
                for tail in self._resampler.flush():
                    self._frame_buffer.append(tail)
                    self._buffered_samples += tail.samples
            frame = self._frame_buffer.popleft() if self._frame_buffer else None
            if frame is not None:
                self._buffered_samples -= frame.samples

        if frame is None:
            # Per-frame sleep keeps silence at the frame cadence even when PTS runs ahead of wall clock after a burst.
            await asyncio.sleep(aiortc.mediastreams.AUDIO_PTIME)
            frame = av.AudioFrame.from_ndarray(
                self._silence, format="s16", layout=self._layout
            )
        elif frame.samples < self._samples_per_frame:
            frame = self._pad_to_full_frame(frame)

        frame.pts = self._timestamp
        frame.sample_rate = self.sample_rate
        frame.time_base = fractions.Fraction(1, self.sample_rate)
        return frame
