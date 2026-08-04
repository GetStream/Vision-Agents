import os
import wave
from pathlib import Path
from typing import AsyncIterator

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import palabra
from vision_agents.plugins.palabra.voices import MAX_SAMPLE_BYTES

load_dotenv()

# Long enough to satisfy Palabra's 30 second minimum when cloned for real.
SAMPLE_SCRIPT = [
    "The harbour wakes slowly every morning, one boat at a time.",
    "The fishermen speak in short sentences, mostly about the weather.",
    "By seven the market is loud and the gulls have taken the high walls.",
    "A woman sells coffee from a cart on the same corner every day.",
    "She knows every regular by the way they hold their cup.",
    "Later the tide turns and the water goes flat and grey.",
    "Children run along the pier, daring each other to look over the edge.",
    "In the evening the boats come back heavier than they left.",
    "Someone always sings on the way in, badly, and nobody minds.",
    "The harbour sleeps again before the town does.",
]


class TestVoices:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PALABRA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            palabra.Voices()

    def test_trailing_slash_is_stripped_from_base_url(self) -> None:
        voices = palabra.Voices(api_key="fake", base_url="https://example.com/")
        assert voices.base_url == "https://example.com"

    async def test_clone_rejects_a_missing_sample(self) -> None:
        voices = palabra.Voices(api_key="fake")
        with pytest.raises(ValueError, match="not found"):
            await voices.clone("Missing", "does-not-exist.wav")

    async def test_clone_rejects_an_unsupported_format(self, tmp_path: Path) -> None:
        sample = tmp_path / "sample.ogg"
        sample.write_bytes(b"not audio")

        voices = palabra.Voices(api_key="fake")
        with pytest.raises(ValueError, match="Unsupported voice sample format"):
            await voices.clone("Wrong format", sample)

    async def test_clone_rejects_an_oversized_sample(self, tmp_path: Path) -> None:
        sample = tmp_path / "sample.wav"
        sample.write_bytes(b"\0" * (MAX_SAMPLE_BYTES + 1))

        voices = palabra.Voices(api_key="fake")
        with pytest.raises(ValueError, match="at most"):
            await voices.clone("Too big", sample)

    def test_ready_reflects_processing_status(self) -> None:
        pending = palabra.ClonedVoice(
            voice_id="v1", name="v", processing_status="pending", errors=[], warnings=[]
        )
        ready = palabra.ClonedVoice(
            voice_id="v1", name="v", processing_status="ready", errors=[], warnings=[]
        )

        assert pending.ready is False
        assert ready.ready is True


@pytest.mark.skipif(
    os.getenv("PALABRA_API_KEY") is None, reason="PALABRA_API_KEY not set"
)
@pytest.mark.integration
class TestVoicesIntegration:
    @pytest.fixture
    async def voices(self) -> AsyncIterator[palabra.Voices]:
        async with palabra.Voices() as voices:
            yield voices

    async def test_limits_reports_the_account_quota(
        self, voices: palabra.Voices
    ) -> None:
        limits = await voices.limits()

        assert limits.limit > 0
        assert limits.remaining == limits.limit - limits.total

    async def test_list_returns_cloned_voices(self, voices: palabra.Voices) -> None:
        listed = await voices.list(page_size=5)

        assert isinstance(listed, list)
        assert all(voice.voice_id for voice in listed)

    @pytest.mark.timeout(300)
    async def test_clone_produces_a_voice_usable_for_synthesis(
        self, voices: palabra.Voices, tmp_path: Path
    ) -> None:
        """Full round trip: record a sample, clone it, speak with it, delete it."""
        sample = tmp_path / "sample.wav"
        tts = palabra.TTS()
        audio = bytearray()
        try:
            for line in SAMPLE_SCRIPT:
                async for chunk in tts.send_iter(line):
                    if chunk.data is not None:
                        audio += chunk.data.samples.tobytes()
            rate = tts.sample_rate
        finally:
            await tts.close()

        with wave.open(str(sample), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio)
        assert len(audio) / 2 / rate > 30

        voice = await voices.clone("Vision Agents test", sample, lang_code="en")
        try:
            assert voice.ready
            assert (await voices.get(voice.voice_id)).voice_id == voice.voice_id

            cloned_tts = palabra.TTS(voice_id=voice.voice_id)
            try:
                chunks = [
                    pcm async for pcm in await cloned_tts.stream_audio("Hello again.")
                ]
            finally:
                await cloned_tts.close()
            assert len(chunks) > 0
        finally:
            await voices.delete(voice.voice_id)

        assert all(v.voice_id != voice.voice_id for v in await voices.list())
