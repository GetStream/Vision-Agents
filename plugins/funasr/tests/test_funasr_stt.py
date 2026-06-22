from vision_agents.plugins import funasr


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
