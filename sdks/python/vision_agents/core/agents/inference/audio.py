from dataclasses import dataclass

from getstream.video.rtc import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.utils.stream import Stream

__all__ = [
    "AudioInputStream",
    "AudioInputChunk",
    "AudioOutputStream",
    "AudioOutputChunk",
    "AudioOutputFlush",
]


@dataclass
class AudioInputChunk:
    data: PcmData
    participant: Participant


@dataclass
class AudioOutputChunk:
    data: PcmData | None = None
    """Pcm audio. Can be empty when only "final=True" is set to signal the end of utterance."""
    final: bool = False


@dataclass
class AudioOutputFlush: ...


class AudioInputStream(Stream[AudioInputChunk]): ...


class AudioOutputStream(Stream[AudioOutputChunk | AudioOutputFlush]):
    """
    A Stream for the output audio that re-chunks it into exact 20ms pieces.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)
        self._carry: PcmData | None = None
        self._chunk_size_ms = 20  # 20ms
        self._chunk_frac = 1000 // self._chunk_size_ms

    @property
    def buffered(self) -> float:
        """Return the amount of seconds of audio pending in the buffer."""
        seconds = sum(
            item.data.duration
            for item in self._items
            if isinstance(item, AudioOutputChunk) and item.data is not None
        )
        if self._carry is not None:
            seconds += self._carry.duration
        return seconds

    def send_nowait(self, item: AudioOutputChunk | AudioOutputFlush) -> None:
        """
        Incoming PcmData of arbitrary length is split into fixed 20ms chunks.
        Samples that don't fill a full 20ms chunk are carried over and prepended
        to the next incoming data.

        When a chunk with ``final=True`` arrives, the
        carry-over is padded to 20ms and emitted automatically.
        """
        if isinstance(item, AudioOutputFlush):
            # Flush item, simply send it downstream.
            super(AudioOutputStream, self).send_nowait(item)
            return

        if item.data is None:
            # No PCM data provided. Still flush any pending carry on a final
            # marker so trailing audio from this utterance is not stranded.
            if item.final:
                self._flush_carry()
            super(AudioOutputStream, self).send_nowait(item)
            return

        pcm = item.data
        chunk_size = pcm.sample_rate // self._chunk_frac

        if self._carry is not None:
            self._carry.append(pcm)
            pcm = self._carry
            self._carry = None

        for pcm_chunk in pcm.chunks(chunk_size):
            # `samples.shape[-1]` is the per-channel sample count for both
            # mono 1D and stereo channel-major 2D arrays. `len(samples)` for
            # 2D returns the channel count, not the sample count.
            if pcm_chunk.samples.shape[-1] < chunk_size:
                self._carry = pcm_chunk
            else:
                super().send_nowait(AudioOutputChunk(data=pcm_chunk))

        if item.final:
            self._flush_carry()
            super().send_nowait(
                AudioOutputChunk(
                    data=PcmData(
                        sample_rate=item.data.sample_rate,
                        format=item.data.format,
                        channels=item.data.channels,
                    ),
                    final=True,
                )
            )

    def _flush_carry(self) -> None:
        if self._carry is not None and self._carry.samples.shape[-1] > 0:
            chunk_size = self._carry.sample_rate // self._chunk_frac
            padded = next(self._carry.chunks(chunk_size, pad_last=True))
            super().send_nowait(AudioOutputChunk(data=padded))
        self._carry = None

    async def flush(self) -> None:
        """
        Write a special "flush" message to the stream to signal downstream consumers
        to empty their buffers, e.g. on interrupt.
        """

        await self.send(AudioOutputFlush())

    def clear(self) -> None:
        super().clear()
        self._carry = None
