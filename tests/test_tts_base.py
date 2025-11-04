import pytest

from vision_agents.core.tts.tts import TTS as BaseTTS
from getstream.video.rtc.track_util import PcmData, AudioFormat
from vision_agents.core.tts.testing import TTSSession


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


async def test_tts_stereo_to_mono_halves_bytes():
    tts = DummyTTSPcmStereoToMono()
    # desired mono, same sample rate
    tts.set_output_format(sample_rate=16000, channels=1)
    session = TTSSession(tts)

    await tts.send("x")
    await tts.events.wait()
    assert len(session.speeches) == 1
    # Original interleaved data length was 400 bytes; mono should be ~200 bytes
    assert 180 <= len(session.speeches[0].to_bytes()) <= 220


async def test_tts_resample_changes_size_reasonably():
    tts = DummyTTSPcmResample()
    # Resample from 16k -> 8k, mono
    tts.set_output_format(sample_rate=8000, channels=1)
    session = TTSSession(tts)

    await tts.send("y")
    await tts.events.wait()
    assert len(session.speeches) == 1
    # Input had 200 samples (400 bytes); at 8k this should be roughly half
    assert 150 <= len(session.speeches[0].to_bytes()) <= 250


async def test_tts_error_emits_and_raises():
    tts = DummyTTSError()
    session = TTSSession(tts)

    with pytest.raises(RuntimeError):
        await tts.send("boom")
    await tts.events.wait()
    assert len(session.errors) >= 1
