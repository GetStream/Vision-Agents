import asyncio

import pytest

from vision_agents.plugins import funasr


class _FakeAutoModel:
    pass


class _ExplodingAutoModel:
    def generate(self, **kwargs):
        raise AssertionError("unexpected transcription bug")


class _FakeSamples:
    def __init__(self, size=0):
        self.size = size


class _FakePcmData:
    def __init__(
        self,
        *,
        samples=None,
        sample_rate=16000,
        channels=1,
        format=None,
        duration_ms=0,
    ):
        self.samples = samples if samples is not None else _FakeSamples()
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.duration_ms = duration_ms

    def resample(self, sample_rate):
        self.sample_rate = sample_rate
        return self

    def to_float32(self):
        return self

    def append(self, audio_data):
        return audio_data


class _ExplodingPcmData:
    samples = _FakeSamples(size=1)

    def resample(self, sample_rate):
        raise AssertionError("unexpected audio bug")


class TestSTT:
    """Unit tests for the FunASR STT plugin (no model loaded)."""

    def test_construct(self):
        """The plugin constructs and exposes the expected defaults without loading a model."""
        stt = funasr.STT()
        assert stt.provider_name == "funasr"
        assert stt.model_id == "iic/SenseVoiceSmall"
        assert stt.language == "auto"
        assert stt.device == "cpu"
        assert stt.use_itn is True

    def test_custom_params(self):
        stt = funasr.STT(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            language="zh",
            device="cuda",
            use_itn=False,
        )
        assert stt.model_id == "FunAudioLLM/Fun-ASR-Nano-2512"
        assert stt.language == "zh"
        assert stt.device == "cuda"
        assert stt.use_itn is False

    def test_process_audio_raises_unexpected_buffer_errors(self):
        stt = funasr.STT(client=_FakeAutoModel())

        with pytest.raises(AssertionError, match="unexpected audio bug"):
            asyncio.run(stt.process_audio(_ExplodingPcmData(), participant=object()))

    def test_process_buffer_raises_unexpected_transcription_errors(self):
        stt = funasr.STT(client=_ExplodingAutoModel())
        pcm_data = _FakePcmData(
            samples=_FakeSamples(size=16000),
            duration_ms=8000,
        )

        with pytest.raises(AssertionError, match="unexpected transcription bug"):
            asyncio.run(stt.process_audio(pcm_data, participant=object()))