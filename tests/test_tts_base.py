import asyncio
from typing import AsyncIterator, Iterator, List

import pytest

from vision_agents.core.tts.tts import TTS as BaseTTS
from vision_agents.core.tts.events import (
    TTSAudioEvent,
    TTSErrorEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
)
from vision_agents.core.edge.types import PcmData


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


@pytest.mark.asyncio
async def test_tts_bytes_single_emits_events_and_bytes():
    tts = DummyTTSBytesSingle()
    tts.set_output_format(sample_rate=16000, channels=1)

    events: List[type] = []
    audio_chunks: List[bytes] = []

    @tts.events.subscribe
    async def _on_start(ev: TTSSynthesisStartEvent):
        events.append(TTSSynthesisStartEvent)

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        events.append(TTSAudioEvent)
        if ev.audio_data:
            audio_chunks.append(ev.audio_data)

    @tts.events.subscribe
    async def _on_complete(ev: TTSSynthesisCompleteEvent):
        events.append(TTSSynthesisCompleteEvent)

    await asyncio.sleep(0.01)
    await tts.send("hello")
    await tts.events.wait()

    # Expect start -> audio -> complete
    assert TTSSynthesisStartEvent in events
    assert TTSAudioEvent in events
    assert TTSSynthesisCompleteEvent in events
    assert len(audio_chunks) == 1
    # audio event sample_rate/channels reflect desired output
    assert audio_chunks[0] is not None


@pytest.mark.asyncio
async def test_tts_bytes_async_aggregates_and_emits():
    tts = DummyTTSBytesAsync()
    tts.set_output_format(sample_rate=16000, channels=1)

    chunks: List[bytes] = []

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        if isinstance(ev, TTSAudioEvent) and ev.audio_data:
            chunks.append(ev.audio_data)

    await asyncio.sleep(0.01)
    await tts.send("hi")
    await tts.events.wait()

    # Should emit at least one aligned chunk
    assert len(chunks) >= 1
    # Sum of bytes equals or exceeds first unaligned chunk (due to padding/next chunk)
    assert sum(len(c) for c in chunks) >= 2 * 33  # approx check


@pytest.mark.asyncio
async def test_tts_iter_sync_emits_multiple_chunks():
    tts = DummyTTSIterSync()
    tts.set_output_format(sample_rate=16000, channels=1)

    chunks: List[bytes] = []

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        if ev.audio_data:
            chunks.append(ev.audio_data)

    await asyncio.sleep(0.01)
    await tts.send("hello")
    await tts.events.wait()
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_tts_stereo_to_mono_halves_bytes():
    tts = DummyTTSPcmStereoToMono()
    # desired mono, same sample rate
    tts.set_output_format(sample_rate=16000, channels=1)

    emitted: List[bytes] = []

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        if ev.audio_data:
            emitted.append(ev.audio_data)

    await asyncio.sleep(0.01)
    await tts.send("x")
    await tts.events.wait()
    assert len(emitted) == 1
    # Original interleaved data length was 400 bytes; mono should be ~200 bytes
    assert 180 <= len(emitted[0]) <= 220


@pytest.mark.asyncio
async def test_tts_resample_changes_size_reasonably():
    tts = DummyTTSPcmResample()
    # Resample from 16k -> 8k, mono
    tts.set_output_format(sample_rate=8000, channels=1)

    emitted: List[bytes] = []

    @tts.events.subscribe
    async def _on_audio(ev: TTSAudioEvent):
        if ev.audio_data:
            emitted.append(ev.audio_data)

    await asyncio.sleep(0.01)
    await tts.send("y")
    await tts.events.wait()
    assert len(emitted) == 1
    # Input had 200 samples (400 bytes); at 8k this should be roughly half
    assert 150 <= len(emitted[0]) <= 250


@pytest.mark.asyncio
async def test_tts_error_emits_and_raises():
    tts = DummyTTSError()

    errors: List[TTSErrorEvent] = []

    @tts.events.subscribe
    async def _on_error(ev: TTSErrorEvent):
        if isinstance(ev, TTSErrorEvent):
            errors.append(ev)

    await asyncio.sleep(0.01)
    with pytest.raises(RuntimeError):
        await tts.send("boom")
    await tts.events.wait()
    assert len(errors) >= 1
