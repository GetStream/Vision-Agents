from typing import AsyncIterator, Iterator

import pytest

from vision_agents.core.tts.tts import TTS as BaseTTS
from vision_agents.core.edge.types import PcmData
from vision_agents.core.tts.testing import TTSSession


class DummyTTSBytesSingle(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> bytes:
        # 16-bit PCM mono (s16), 100 samples -> 200 bytes
        self._native_sample_rate = 16000
        self._native_channels = 1
        return b"\x00\x00" * 100

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSBytesAsync(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> AsyncIterator[bytes]:
        self._native_sample_rate = 16000
        self._native_channels = 1

        async def _agen():
            # Unaligned chunk sizes to test aggregator
            yield b"\x00\x00" * 33 + b"\x00"  # odd size
            yield b"\x00\x00" * 10

        return _agen()

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSIterSync(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> Iterator[bytes]:
        self._native_sample_rate = 16000
        self._native_channels = 1
        return iter([b"\x00\x00" * 50, b"\x00\x00" * 25])

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSPcmStereoToMono(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        # 2 channels interleaved: 100 frames (per channel) -> 200 samples -> 400 bytes
        frames = b"\x01\x00\x01\x00" * 100  # L(1), R(1)
        pcm = PcmData.from_bytes(frames, sample_rate=16000, channels=2, format="s16")
        return pcm

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSPcmResample(BaseTTS):
    async def stream_audio(self, text: str, *_, **__) -> PcmData:
        # 16k mono, 200 samples (duration = 200/16000 s)
        data = b"\x00\x00" * 200
        pcm = PcmData.from_bytes(data, sample_rate=16000, channels=1, format="s16")
        return pcm

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


class DummyTTSError(BaseTTS):
    async def stream_audio(self, text: str, *_, **__):
        raise RuntimeError("boom")

    async def stop_audio(self) -> None:  # pragma: no cover - noop
        return None


async def test_tts_bytes_single_emits_events_and_bytes():
    tts = DummyTTSBytesSingle()
    tts.set_output_format(sample_rate=16000, channels=1)
    session = TTSSession(tts)

    await tts.send("hello")
    await tts.events.wait()
    result = await session.wait_for_result(timeout=1.0)

    assert result.started
    assert result.completed
    assert len(session.speeches) == 1
    assert session.speeches[0] is not None


async def test_tts_bytes_async_aggregates_and_emits():
    tts = DummyTTSBytesAsync()
    tts.set_output_format(sample_rate=16000, channels=1)
    session = TTSSession(tts)

    await tts.send("hi")
    await tts.events.wait()

    assert len(session.speeches) >= 1
    assert sum(len(c) for c in session.speeches) >= 2 * 33  # approx check


async def test_tts_iter_sync_emits_multiple_chunks():
    tts = DummyTTSIterSync()
    tts.set_output_format(sample_rate=16000, channels=1)
    session = TTSSession(tts)

    await tts.send("hello")
    await tts.events.wait()
    assert len(session.speeches) >= 2


async def test_tts_stereo_to_mono_halves_bytes():
    tts = DummyTTSPcmStereoToMono()
    # desired mono, same sample rate
    tts.set_output_format(sample_rate=16000, channels=1)
    session = TTSSession(tts)

    await tts.send("x")
    await tts.events.wait()
    assert len(session.speeches) == 1
    # Original interleaved data length was 400 bytes; mono should be ~200 bytes
    assert 180 <= len(session.speeches[0]) <= 220


async def test_tts_resample_changes_size_reasonably():
    tts = DummyTTSPcmResample()
    # Resample from 16k -> 8k, mono
    tts.set_output_format(sample_rate=8000, channels=1)
    session = TTSSession(tts)

    await tts.send("y")
    await tts.events.wait()
    assert len(session.speeches) == 1
    # Input had 200 samples (400 bytes); at 8k this should be roughly half
    assert 150 <= len(session.speeches[0]) <= 250


async def test_tts_error_emits_and_raises():
    tts = DummyTTSError()
    session = TTSSession(tts)

    with pytest.raises(RuntimeError):
        await tts.send("boom")
    await tts.events.wait()
    assert len(session.errors) >= 1
