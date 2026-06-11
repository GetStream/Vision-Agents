import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from getstream.video.rtc.track_util import PcmData

from ..edge.types import Participant
from .audio_queue import AudioQueue

logger = logging.getLogger(__name__)

AudioInputSender = Callable[[PcmData, Participant | None], Awaitable[None]]


@dataclass(slots=True)
class AudioInputPacingConfig:
    """Configuration for steady realtime audio input forwarding."""

    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: float = 20.0
    startup_buffer_ms: float = 300.0
    max_buffer_ms: float = 1500.0
    silence_when_empty: bool = False

    @classmethod
    def virtual_microphone(cls) -> "AudioInputPacingConfig":
        return cls(startup_buffer_ms=500.0, silence_when_empty=True)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if self.startup_buffer_ms < self.chunk_ms:
            raise ValueError("startup_buffer_ms must be at least chunk_ms")
        if self.max_buffer_ms < self.startup_buffer_ms:
            raise ValueError("max_buffer_ms must be at least startup_buffer_ms")


class AudioInputPacer:
    """Buffers input audio and forwards it at a stable wall-clock cadence."""

    def __init__(
        self,
        config: AudioInputPacingConfig,
        send: AudioInputSender,
        name: str = "audio_input_pacer",
    ) -> None:
        self.config = config
        self._send = send
        self._name = name
        self._queue = AudioQueue(buffer_limit_ms=int(config.max_buffer_ms))
        self._task: asyncio.Task[None] | None = None
        self._streaming = False
        self._participant: Participant | None = None
        self.chunks_sent = 0
        self.silence_chunks_sent = 0
        self.pauses = 0

    async def push(self, pcm: PcmData, participant: Participant | None) -> None:
        """Add audio to the pacing buffer, resampling to the configured format."""
        self.start()
        self._participant = participant
        audio = pcm.resample(
            target_sample_rate=self.config.sample_rate,
            target_channels=self.config.channels,
        ).to_int16()
        if audio.duration_ms > self.config.max_buffer_ms:
            audio = audio.tail(self.config.max_buffer_ms / 1000)
        audio.participant = participant
        await self._queue.put(audio)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def clear(self) -> None:
        self._streaming = False
        self._participant = None
        self._queue.clear()

    async def close(self) -> None:
        self.clear()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def buffered_ms(self) -> float:
        return float(self._queue.get_buffer_info()["current_duration_ms"])

    async def _run(self) -> None:
        interval = self.config.chunk_ms / 1000
        loop = asyncio.get_running_loop()
        next_t = loop.time()
        try:
            while True:
                next_t += interval
                delay = next_t - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -1.0:
                    next_t = loop.time()

                if not self._ready_to_send():
                    continue

                chunk = await self._next_chunk()
                if chunk is None:
                    continue

                try:
                    await self._send(chunk, chunk.participant)
                except Exception:
                    logger.exception("%s send failed", self._name)
                    continue

                self.chunks_sent += 1

                if self.chunks_sent % 500 == 0:
                    logger.info(
                        "%s forwarded %d chunks, silence=%d, pauses=%d, buffered=%.0fms",
                        self._name,
                        self.chunks_sent,
                        self.silence_chunks_sent,
                        self.pauses,
                        self.buffered_ms(),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("%s stopped unexpectedly", self._name)

    async def _next_chunk(self) -> PcmData | None:
        if self.buffered_ms() >= self.config.chunk_ms:
            try:
                return await self._queue.get_duration(self.config.chunk_ms)
            except asyncio.QueueEmpty:
                return None

        if not self.config.silence_when_empty:
            return None

        samples = int(self.config.sample_rate * self.config.chunk_ms / 1000)
        silence = PcmData.from_bytes(
            b"\x00" * samples * self.config.channels * 2,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )
        silence.participant = self._participant
        self.silence_chunks_sent += 1
        return silence

    def _ready_to_send(self) -> bool:
        buffered_ms = self.buffered_ms()
        if self._streaming:
            if self.config.silence_when_empty or buffered_ms >= self.config.chunk_ms:
                return True
            self._streaming = False
            self.pauses += 1
            return False

        if buffered_ms >= self.config.startup_buffer_ms:
            self._streaming = True
            return True
        return False
