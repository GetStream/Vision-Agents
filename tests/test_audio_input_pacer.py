import asyncio
from collections.abc import AsyncIterator

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.realtime import Realtime
from vision_agents.core.utils.audio_input_pacer import (
    AudioInputPacer,
    AudioInputPacingConfig,
)
from vision_agents.core.utils.audio_queue import AudioQueue

from tests.base_test import BaseTest


def _pcm(value: int, duration_ms: float, sample_rate: int = 16000) -> PcmData:
    samples = np.full(int(sample_rate * duration_ms / 1000), value, dtype=np.int16)
    return PcmData(
        samples=samples,
        sample_rate=sample_rate,
        format=AudioFormat.S16,
        channels=1,
    )


async def _wait_until(predicate, timeout: float = 0.25) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("condition was not met before timeout")


class _RealtimeForPacingTest(Realtime):
    provider_name = "test_realtime"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sent: list[PcmData] = []

    async def connect(self):
        self._on_connected()

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        yield LLMResponseFinal()

    async def simple_audio_response(
        self, pcm: PcmData, participant: Participant | None = None
    ):
        self.sent.append(pcm)

    async def watch_video_track(self, *args, **kwargs) -> None:
        pass

    async def close(self) -> None:
        await self._close_input_audio_pacer()
        self._on_disconnected()


class _ClearingAudioQueue(AudioQueue):
    def __init__(self, pacer: AudioInputPacer, chunk: PcmData) -> None:
        super().__init__(buffer_limit_ms=100)
        self._pacer = pacer
        self._chunk = chunk
        self.was_cleared = False

    def clear(self) -> None:
        self.was_cleared = True

    def get_buffer_info(self) -> dict:
        return {"current_duration_ms": 0 if self.was_cleared else 10}

    async def get_duration(self, duration_ms: float) -> PcmData:
        self._pacer.clear()
        return self._chunk


class TestAudioInputPacer(BaseTest):
    async def test_realtime_process_audio_uses_configured_pacer(self):
        participant = Participant(original=None, user_id="u", id="u")
        realtime = _RealtimeForPacingTest(
            input_audio_pacing=AudioInputPacingConfig(
                chunk_ms=5,
                startup_buffer_ms=10,
                max_buffer_ms=100,
            )
        )
        await realtime.connect()

        try:
            await realtime.process_audio(_pcm(1, 5), participant)
            await asyncio.sleep(0.02)
            assert realtime.sent == []

            await realtime.process_audio(_pcm(2, 5), participant)
            await _wait_until(lambda: len(realtime.sent) >= 1)
            assert realtime.sent[0].duration_ms == 5
        finally:
            await realtime.close()

    async def test_virtual_microphone_fills_silence_after_prime(self):
        sent: list[PcmData] = []

        async def send(pcm: PcmData, participant):
            sent.append(pcm)

        pacer = AudioInputPacer(
            AudioInputPacingConfig(
                chunk_ms=5,
                startup_buffer_ms=10,
                max_buffer_ms=100,
                silence_when_empty=True,
            ),
            send,
        )

        try:
            await pacer.push(_pcm(3, 10), None)
            await _wait_until(lambda: len(sent) >= 3)
            assert np.all(sent[0].samples == 3)
            assert np.all(sent[1].samples == 3)
            assert any(np.all(chunk.samples == 0) for chunk in sent[2:])
            assert pacer.silence_chunks_sent > 0

            pacer.clear()
            sent_after_clear = len(sent)
            await asyncio.sleep(0.02)
            assert len(sent) == sent_after_clear
        finally:
            await pacer.close()

    async def test_clear_during_chunk_fetch_drops_stale_chunk(self):
        sent: list[PcmData] = []

        async def send(pcm: PcmData, participant):
            sent.append(pcm)

        pacer = AudioInputPacer(
            AudioInputPacingConfig(
                chunk_ms=5,
                startup_buffer_ms=5,
                max_buffer_ms=100,
            ),
            send,
        )
        queue = _ClearingAudioQueue(pacer, _pcm(4, 5))
        pacer._queue = queue

        try:
            pacer.start()
            await _wait_until(lambda: queue.was_cleared)
            await asyncio.sleep(0.02)
            assert sent == []
            assert pacer.chunks_sent == 0
        finally:
            await pacer.close()
