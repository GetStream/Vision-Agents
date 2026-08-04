import asyncio
import json
import os
from typing import AsyncIterator

import pytest
from dotenv import load_dotenv
from vision_agents.plugins import palabra
from vision_agents.plugins.palabra.tts import MAX_TEXT_LENGTH, _split_text

load_dotenv()


class TestPalabraTTS:
    def test_defaults(self) -> None:
        tts = palabra.TTS(api_key="fake")
        assert tts.voice_id == "default_low"
        assert tts.language == "en"
        assert tts.model == "auto"
        assert tts.sample_rate == 24000
        assert tts.streaming is True

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PALABRA_API_KEY", raising=False)
        with pytest.raises(ValueError):
            palabra.TTS()

    def test_out_of_range_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            palabra.TTS(api_key="fake", sample_rate=96000)

    def test_out_of_range_speed_raises(self) -> None:
        with pytest.raises(ValueError):
            palabra.TTS(api_key="fake", speed=3.0)

    def test_init_message_requests_pcm_with_configured_voice(self) -> None:
        tts = palabra.TTS(
            api_key="fake",
            voice_id="default_high",
            language="de",
            sample_rate=16000,
            speed=1.2,
            deaccent_strength=0.5,
        )
        message = json.loads(tts._init_message)

        assert message == {
            "type": "init",
            "language": "de",
            "model": "auto",
            "voice_options": {
                "voice_id": "default_high",
                "speed": 1.2,
                "deaccent_strength": 0.5,
            },
            "output": {"format": "pcm", "sample_rate": 16000},
        }

    def test_init_message_omits_unset_voice_options(self) -> None:
        tts = palabra.TTS(api_key="fake")
        message = json.loads(tts._init_message)

        assert message["voice_options"] == {"voice_id": "default_low"}

    def test_short_text_is_sent_as_a_single_message(self) -> None:
        assert _split_text("Hello there.") == ["Hello there."]

    def test_long_text_is_split_on_word_boundaries(self) -> None:
        text = " ".join(["word"] * 600)
        chunks = _split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= MAX_TEXT_LENGTH for chunk in chunks)
        assert " ".join(chunks) == text

    def test_long_text_without_spaces_is_split_at_the_limit(self) -> None:
        text = "a" * (MAX_TEXT_LENGTH + 10)
        chunks = _split_text(text)

        assert chunks == ["a" * MAX_TEXT_LENGTH, "a" * 10]


@pytest.mark.skipif(
    os.getenv("PALABRA_API_KEY") is None, reason="PALABRA_API_KEY not set"
)
@pytest.mark.integration
class TestPalabraTTSIntegration:
    @pytest.fixture
    async def tts(self) -> AsyncIterator[palabra.TTS]:
        tts = palabra.TTS()
        yield tts
        await tts.close()

    async def test_convert_text_to_audio(self, tts: palabra.TTS) -> None:
        out = []
        async for item in tts.send_iter("Hello from Palabra AI."):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[0].data.sample_rate == tts.sample_rate
        assert out[-1].final

    async def test_synthesizes_text_longer_than_one_message(
        self, tts: palabra.TTS
    ) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 40
        assert len(text) > MAX_TEXT_LENGTH

        chunks = [pcm async for pcm in await tts.stream_audio(text)]

        assert len(chunks) > 0

    async def test_stop_audio_keeps_session_usable(self, tts: palabra.TTS) -> None:
        long_text = (
            "This is a fairly long sentence that the server should synthesize "
            "across many audio chunks before it completes. " * 4
        )

        chunks = []
        async for pcm in await tts.stream_audio(long_text):
            chunks.append(pcm)
            if len(chunks) == 2:
                await tts.stop_audio()

        assert len(chunks) >= 2

        follow_up = [pcm async for pcm in await tts.stream_audio("Hello again.")]
        assert len(follow_up) > 0

    async def test_stop_audio_releases_a_reader_waiting_on_the_server(
        self, tts: palabra.TTS
    ) -> None:
        """Barge-in arrives on a different task than the one reading audio.

        Palabra does not acknowledge `cancel`, so the reader has to be released
        by the plugin itself or the agent stays stuck mid-utterance.
        """
        long_text = (
            "This is a fairly long sentence that the server should synthesize "
            "across many audio chunks before it completes. " * 4
        )

        async def drain() -> int:
            count = 0
            async for _ in await tts.stream_audio(long_text):
                count += 1
            return count

        reader = asyncio.ensure_future(drain())
        await asyncio.sleep(0.5)
        await tts.stop_audio()

        await asyncio.wait_for(reader, timeout=5)

        follow_up = [pcm async for pcm in await tts.stream_audio("Hello again.")]
        assert len(follow_up) > 0

    async def test_idle_timeout_ends_a_generation_that_never_produces_audio(
        self,
    ) -> None:
        """A wedged generation must not stall the pipeline forever.

        Driven with an unreachably short idle timeout so the guard trips on the
        first read instead of waiting on a real server fault.
        """
        tts = palabra.TTS(idle_timeout=0.001)
        try:
            chunks = [pcm async for pcm in await tts.stream_audio("Hello there.")]
            assert chunks == []

            tts._idle_timeout = 5.0
            recovered = [pcm async for pcm in await tts.stream_audio("Hello again.")]
            assert len(recovered) > 0
        finally:
            await tts.close()

    async def test_reuses_a_single_connection_across_utterances(
        self, tts: palabra.TTS
    ) -> None:
        async for _ in await tts.stream_audio("First utterance."):
            pass
        websocket = tts._websocket

        async for _ in await tts.stream_audio("Second utterance."):
            pass

        assert tts._websocket is websocket
