from typing import AsyncIterator, Iterator

import pytest

from getstream.video.rtc.track_util import AudioFormat, PcmData

from vision_agents.core.tts.events import (
    TTSAudioEvent,
    TTSSynthesisCompleteEvent,
    TTSSynthesisStartEvent,
)
from vision_agents.core.tts.testing import TTSSession
from vision_agents.core.tts.tts import TTS as BaseTTS


class DummyTTSPcmStereoToMono(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        # 2 channels interleaved: 100 frames (per channel) -> 200 samples -> 400 bytes
        frames = b"\x01\x00\x01\x00" * 100  # L(1), R(1)
        pcm = PcmData.from_bytes(
            frames, sample_rate=16000, channels=2, format=AudioFormat.S16
        )
        return pcm

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSPcmResample(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        # 16k mono, 200 samples (duration = 200/16000 s)
        data = b"\x00\x00" * 200
        pcm = PcmData.from_bytes(
            data, sample_rate=16000, channels=1, format=AudioFormat.S16
        )
        return pcm

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSError(BaseTTS):
    async def stream_audio(self, text: str, *_, **__):
        raise RuntimeError("boom")

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


def _make_pcm(n_samples: int = 100, sample_rate: int = 16000) -> PcmData:
    data = b"\x01\x00" * n_samples
    return PcmData.from_bytes(
        data, sample_rate=sample_rate, channels=1, format=AudioFormat.S16
    )


class DummyTTSAsyncIter(BaseTTS):
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


class DummyTTSSyncIter(BaseTTS):
    """stream_audio returns a sync iterator of PcmData chunks."""

    def __init__(self, chunks: list[PcmData]):
        super().__init__()
        self._chunks = chunks

    async def stream_audio(self, text: str, *_, **__) -> Iterator[PcmData]:
        return iter(self._chunks)

    async def stop_audio(self) -> None:
        return None


class DummyTTSAsyncIterBadType(BaseTTS):
    """stream_audio yields non-PcmData from an async iterator."""

    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[PcmData]:
        async def _gen():
            yield b"not-pcm-data"

        return _gen()

    async def stop_audio(self) -> None:
        return None


class DummyTTSSyncIterBadType(BaseTTS):
    """stream_audio yields non-PcmData from a sync iterator."""

    async def stream_audio(self, text: str, *_, **__):
        return iter([b"not-pcm-data"])

    async def stop_audio(self) -> None:
        return None


class DummyTTSUnsupportedReturn(BaseTTS):
    """stream_audio returns raw bytes (unsupported)."""

    async def stream_audio(self, text: str, *_, **__) -> bytes:
        return b"\x00\x00" * 100

    async def stop_audio(self) -> None:
        return None


def _collect_audio_events(tts: BaseTTS) -> list[TTSAudioEvent]:
    collected: list[TTSAudioEvent] = []

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        collected.append(ev)

    return collected


def _collect_complete_events(tts: BaseTTS) -> list[TTSSynthesisCompleteEvent]:
    collected: list[TTSSynthesisCompleteEvent] = []

    @tts.events.subscribe
    async def _on_complete(ev: TTSSynthesisCompleteEvent):
        collected.append(ev)

    return collected


def _collect_start_events(tts: BaseTTS) -> list[TTSSynthesisStartEvent]:
    collected: list[TTSSynthesisStartEvent] = []

    @tts.events.subscribe
    async def _on_start(ev: TTSSynthesisStartEvent):
        collected.append(ev)

    return collected


class TestTTS:
    async def test_stereo_to_mono_halves_bytes(self):
        tts = DummyTTSPcmStereoToMono()
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)

        await tts.send("x")
        await tts.events.wait(timeout=1.0)
        assert len(session.speeches) == 1
        assert 180 <= len(session.speeches[0].to_bytes()) <= 220

    async def test_resample_changes_size_reasonably(self):
        tts = DummyTTSPcmResample()
        tts.set_output_format(sample_rate=8000, channels=1)
        session = TTSSession(tts)

        await tts.send("y")
        await tts.events.wait(timeout=1.0)
        assert len(session.speeches) == 1
        assert 150 <= len(session.speeches[0].to_bytes()) <= 250

    async def test_error_emits_and_raises(self):
        tts = DummyTTSError()
        session = TTSSession(tts)

        with pytest.raises(RuntimeError):
            await tts.send("boom")
        await tts.events.wait(timeout=1.0)
        assert len(session.errors) >= 1

    async def test_async_iter_emits_per_chunk_events(self):
        chunks = [_make_pcm(100), _make_pcm(200), _make_pcm(150)]
        tts = DummyTTSAsyncIter(chunks)
        tts.set_output_format(sample_rate=16000, channels=1)
        audio_events = _collect_audio_events(tts)

        await tts.send("hello")
        await tts.events.wait(timeout=1.0)

        assert len(audio_events) == 4
        for i in range(3):
            assert audio_events[i].chunk_index == i
            assert audio_events[i].is_final_chunk is False
            assert audio_events[i].data is not None

        final = audio_events[3]
        assert final.chunk_index == 3
        assert final.is_final_chunk is True
        assert final.data is None

    async def test_sync_iter_emits_per_chunk_events(self):
        chunks = [_make_pcm(100), _make_pcm(200)]
        tts = DummyTTSSyncIter(chunks)
        tts.set_output_format(sample_rate=16000, channels=1)
        audio_events = _collect_audio_events(tts)

        await tts.send("hello")
        await tts.events.wait(timeout=1.0)

        assert len(audio_events) == 3
        assert audio_events[0].data is not None
        assert audio_events[1].data is not None
        assert audio_events[2].is_final_chunk is True
        assert audio_events[2].data is None

    async def test_single_pcm_fast_path_is_final(self):
        tts = DummyTTSPcmResample()
        tts.set_output_format(sample_rate=16000, channels=1)
        audio_events = _collect_audio_events(tts)

        await tts.send("fast")
        await tts.events.wait(timeout=1.0)

        assert len(audio_events) == 1
        assert audio_events[0].chunk_index == 0
        assert audio_events[0].is_final_chunk is True
        assert audio_events[0].data is not None

    async def test_streaming_completion_event_has_correct_chunk_count(self):
        chunks = [_make_pcm(100), _make_pcm(100), _make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks)
        tts.set_output_format(sample_rate=16000, channels=1)
        complete_events = _collect_complete_events(tts)

        await tts.send("count")
        await tts.events.wait(timeout=1.0)

        assert len(complete_events) == 1
        assert complete_events[0].chunk_count == 3

    async def test_streaming_emits_start_and_complete(self):
        chunks = [_make_pcm(100)]
        tts = DummyTTSAsyncIter(chunks)
        start_events = _collect_start_events(tts)
        complete_events = _collect_complete_events(tts)

        await tts.send("lifecycle")
        await tts.events.wait(timeout=1.0)

        assert len(start_events) == 1
        assert start_events[0].text == "lifecycle"
        assert len(complete_events) == 1
        assert complete_events[0].synthesis_id == start_events[0].synthesis_id

    async def test_async_iter_bad_type_raises_and_emits_error(self):
        tts = DummyTTSAsyncIterBadType()
        session = TTSSession(tts)

        with pytest.raises(TypeError, match="stream_audio must yield PcmData"):
            await tts.send("bad")
        await tts.events.wait(timeout=1.0)
        assert len(session.errors) >= 1

    async def test_sync_iter_bad_type_raises_and_emits_error(self):
        tts = DummyTTSSyncIterBadType()
        session = TTSSession(tts)

        with pytest.raises(TypeError, match="stream_audio must yield PcmData"):
            await tts.send("bad")
        await tts.events.wait(timeout=1.0)
        assert len(session.errors) >= 1

    async def test_unsupported_return_type_raises_and_emits_error(self):
        tts = DummyTTSUnsupportedReturn()
        session = TTSSession(tts)

        with pytest.raises(TypeError, match="Unsupported return type"):
            await tts.send("bad")
        await tts.events.wait(timeout=1.0)
        assert len(session.errors) >= 1

    async def test_streaming_resamples_each_chunk(self):
        chunks = [_make_pcm(160, sample_rate=16000), _make_pcm(160, sample_rate=16000)]
        tts = DummyTTSAsyncIter(chunks)
        tts.set_output_format(sample_rate=48000, channels=1)
        audio_events = _collect_audio_events(tts)

        await tts.send("resample")
        await tts.events.wait(timeout=1.0)

        for ev in audio_events:
            if ev.data is not None:
                assert ev.data.sample_rate == 48000

    async def test_empty_async_iter_no_final_sentinel(self):
        tts = DummyTTSAsyncIter([])
        audio_events = _collect_audio_events(tts)
        complete_events = _collect_complete_events(tts)

        await tts.send("empty")
        await tts.events.wait(timeout=1.0)

        assert len(audio_events) == 0
        assert len(complete_events) == 1
        assert complete_events[0].chunk_count == 0
