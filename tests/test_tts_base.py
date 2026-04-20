from typing import AsyncIterator, Iterator

import pytest
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.tts.events import (
    TTSErrorEvent,
    TTSSynthesisCompleteEvent,
    TTSSynthesisStartEvent,
)
from vision_agents.core.tts.tts import TTS, TTSOutputChunk


class DummyTTSPcm(TTS):
    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        # 16k mono, 200 samples
        data = b"\x00\x00" * 200
        pcm = PcmData.from_bytes(
            data, sample_rate=16000, channels=1, format=AudioFormat.S16
        )
        return pcm

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSError(TTS):
    async def stream_audio(self, text: str, *_, **__):
        raise RuntimeError("boom")

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


def _make_pcm(n_samples: int = 100, sample_rate: int = 16000) -> PcmData:
    data = b"\x01\x00" * n_samples
    return PcmData.from_bytes(
        data, sample_rate=sample_rate, channels=1, format=AudioFormat.S16
    )


class DummyTTSAsyncIter(TTS):
    """stream_audio returns an async iterator of PcmData chunks."""

    def __init__(self, chunks: list[PcmData]):
        super().__init__()
        self._chunks = chunks

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        async def _gen() -> AsyncIterator[PcmData]:
            for chunk in self._chunks:
                yield chunk

        return _gen()

    async def stop_audio(self) -> None:
        return None


class DummyTTSSyncIter(TTS):
    """stream_audio returns a sync iterator of PcmData chunks."""

    def __init__(self, chunks: list[PcmData]):
        super().__init__()
        self._chunks = chunks

    async def stream_audio(self, text: str, *_, **__) -> Iterator[PcmData]:
        return iter(self._chunks)

    async def stop_audio(self) -> None:
        return None


class DummyTTSAsyncIterBadType(TTS):
    """stream_audio yields non-PcmData from an async iterator."""

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        async def _gen():
            yield b"not-pcm-data"

        return _gen()

    async def stop_audio(self) -> None:
        return None


class DummyTTSSyncIterBadType(TTS):
    """stream_audio yields non-PcmData from a sync iterator."""

    async def stream_audio(self, text: str, *_, **__):
        return iter([b"not-pcm-data"])

    async def stop_audio(self) -> None:
        return None


class DummyTTSUnsupportedReturn(TTS):
    """stream_audio returns raw bytes (unsupported)."""

    async def stream_audio(self, text: str, *_, **__) -> bytes:
        return b"\x00\x00" * 100

    async def stop_audio(self) -> None:
        return None


def _collect_complete_events(tts: TTS) -> list[TTSSynthesisCompleteEvent]:
    collected: list[TTSSynthesisCompleteEvent] = []

    @tts.events.subscribe
    async def _on_complete(ev: TTSSynthesisCompleteEvent):
        collected.append(ev)

    return collected


def _collect_start_events(tts: TTS) -> list[TTSSynthesisStartEvent]:
    collected: list[TTSSynthesisStartEvent] = []

    @tts.events.subscribe
    async def _on_start(ev: TTSSynthesisStartEvent):
        collected.append(ev)

    return collected


def _collect_error_events(tts: TTS) -> list[TTSErrorEvent]:
    collected: list[TTSErrorEvent] = []

    @tts.events.subscribe
    async def _on_error(ev: TTSErrorEvent):
        collected.append(ev)

    return collected


async def _drain(it: AsyncIterator[TTSOutputChunk]) -> list[TTSOutputChunk]:
    return [c async for c in it]


class TestTTS:
    async def test_send_iter_single_pcm_is_final(self):
        tts = DummyTTSPcm()

        chunks = await _drain(tts.send_iter("fast"))

        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].final is True
        assert chunks[0].data is not None

    async def test_send_iter_async_iter_yields_chunks_plus_sentinel(self):
        chunks_in = [_make_pcm(100), _make_pcm(200), _make_pcm(150)]
        tts = DummyTTSAsyncIter(chunks_in)

        chunks = await _drain(tts.send_iter("hello"))

        assert len(chunks) == 4
        for i in range(3):
            assert chunks[i].index == i
            assert chunks[i].final is False
            assert chunks[i].data is not None

        sentinel = chunks[3]
        assert sentinel.index == 3
        assert sentinel.final is True
        assert sentinel.data is None

    async def test_send_iter_sync_iter_yields_chunks_plus_sentinel(self):
        chunks_in = [_make_pcm(100), _make_pcm(200)]
        tts = DummyTTSSyncIter(chunks_in)

        chunks = await _drain(tts.send_iter("hello"))

        assert len(chunks) == 3
        assert chunks[0].data is not None
        assert chunks[1].data is not None
        assert chunks[2].final is True
        assert chunks[2].data is None

    async def test_send_iter_empty_async_iter_no_sentinel(self):
        tts = DummyTTSAsyncIter([])
        complete_events = _collect_complete_events(tts)

        chunks = await _drain(tts.send_iter("empty"))
        await tts.events.wait(timeout=1.0)

        assert chunks == []
        assert len(complete_events) == 1
        assert complete_events[0].chunk_count == 0

    async def test_send_iter_emits_start_and_complete(self):
        chunks_in = [_make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks_in)
        start_events = _collect_start_events(tts)
        complete_events = _collect_complete_events(tts)

        chunks = await _drain(tts.send_iter("lifecycle"))
        await tts.events.wait(timeout=1.0)

        assert len(start_events) == 1
        assert start_events[0].text == "lifecycle"
        assert len(complete_events) == 1
        assert complete_events[0].synthesis_id == start_events[0].synthesis_id
        assert chunks[0].synthesis_id == start_events[0].synthesis_id

    async def test_send_iter_complete_event_chunk_count(self):
        chunks_in = [_make_pcm(100), _make_pcm(100), _make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks_in)
        complete_events = _collect_complete_events(tts)

        await _drain(tts.send_iter("count"))
        await tts.events.wait(timeout=1.0)

        assert len(complete_events) == 1
        assert complete_events[0].chunk_count == 3

    async def test_send_iter_error_raises(self):
        tts = DummyTTSError()

        with pytest.raises(RuntimeError):
            await _drain(tts.send_iter("boom"))

    async def test_send_iter_async_iter_bad_type_raises(self):
        tts = DummyTTSAsyncIterBadType()

        with pytest.raises(TypeError, match="stream_audio must yield PcmData"):
            await _drain(tts.send_iter("bad"))

    async def test_send_iter_sync_iter_bad_type_raises(self):
        tts = DummyTTSSyncIterBadType()

        with pytest.raises(TypeError, match="stream_audio must yield PcmData"):
            await _drain(tts.send_iter("bad"))

    async def test_send_iter_unsupported_return_type_raises(self):
        tts = DummyTTSUnsupportedReturn()

        with pytest.raises(TypeError, match="Unsupported return type"):
            await _drain(tts.send_iter("bad"))

    async def test_send_iter_synthesis_id_consistent_across_chunks(self):
        chunks_in = [_make_pcm(100), _make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks_in)

        chunks = await _drain(tts.send_iter("hello"))

        sid = chunks[0].synthesis_id
        assert sid
        assert all(c.synthesis_id == sid for c in chunks)
        assert all(c.text == "hello" for c in chunks)

    async def test_send_iter_interrupt_stops_iteration(self):
        chunks_in = [_make_pcm(100), _make_pcm(100), _make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks_in)

        collected: list[TTSOutputChunk] = []
        async for chunk in tts.send_iter("interruptible"):
            collected.append(chunk)
            if len(collected) == 1:
                await tts.interrupt()

        assert len(collected) == 1
