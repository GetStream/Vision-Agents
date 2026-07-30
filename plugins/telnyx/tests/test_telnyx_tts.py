"""Tests for the Telnyx TTS plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.plugins.telnyx import TTS
from vision_agents.plugins.telnyx.tts import _Id3Stripper

load_dotenv()


def id3_tag(payload_size: int) -> bytes:
    """Build an ID3v2 tag header plus a body of ``payload_size`` bytes."""
    synchsafe = bytes(
        [
            (payload_size >> 21) & 0x7F,
            (payload_size >> 14) & 0x7F,
            (payload_size >> 7) & 0x7F,
            payload_size & 0x7F,
        ]
    )
    return b"ID3\x04\x00\x00" + synchsafe + b"\xaa" * payload_size


class TestId3Stripper:
    """Unit tests for the streaming ID3v2 tag stripper."""

    def test_untagged_data_passes_through(self):
        stripper = _Id3Stripper()
        assert stripper.feed(b"\xff\xf3audio") == b"\xff\xf3audio"

    def test_leading_tag_removed(self):
        stripper = _Id3Stripper()
        assert stripper.feed(id3_tag(34) + b"\xff\xf3audio") == b"\xff\xf3audio"

    def test_zero_length_tag_removed(self):
        stripper = _Id3Stripper()
        assert stripper.feed(id3_tag(0) + b"\xff\xf3") == b"\xff\xf3"

    def test_tag_body_spanning_frames(self):
        stripper = _Id3Stripper()
        blob = id3_tag(40) + b"\xff\xf3audio"
        assert stripper.feed(blob[:20]) == b""
        assert stripper.feed(blob[20:]) == b"\xff\xf3audio"

    def test_tag_header_spanning_frames(self):
        stripper = _Id3Stripper()
        blob = id3_tag(12) + b"\xff\xf3audio"
        assert stripper.feed(blob[:4]) == b""
        assert stripper.feed(blob[4:]) == b"\xff\xf3audio"

    def test_tag_at_head_of_later_frame(self):
        stripper = _Id3Stripper()
        assert stripper.feed(b"\xff\xf3first") == b"\xff\xf3first"
        assert stripper.feed(id3_tag(8) + b"\xff\xf3second") == b"\xff\xf3second"

    def test_id3_bytes_inside_audio_are_kept(self):
        stripper = _Id3Stripper()
        audio = b"\xff\xf3 padding ID3 more audio"
        assert stripper.feed(audio) == audio

    def test_tag_consuming_whole_frame(self):
        stripper = _Id3Stripper()
        blob = id3_tag(100) + b"\xff\xf3audio"
        assert stripper.feed(blob[:50]) == b""
        assert stripper.feed(blob[50:]) == b"\xff\xf3audio"


class TestTelnyxTTS:
    """Unit tests for Telnyx TTS configuration."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            TTS()

    async def test_default_configuration(self):
        tts = TTS(api_key="KEY_test")
        assert tts.voice == "Telnyx.KokoroTTS.af_heart"
        assert tts.provider_name == "telnyx"

    async def test_custom_voice(self):
        tts = TTS(api_key="KEY_test", voice="AWS.Polly.Danielle-Neural")
        assert tts.voice == "AWS.Polly.Danielle-Neural"


@pytest.mark.skipif(not os.getenv("TELNYX_API_KEY"), reason="TELNYX_API_KEY not set")
@pytest.mark.integration
class TestTelnyxTTSIntegration:
    """Integration tests against the real Telnyx streaming TTS."""

    @pytest.fixture
    async def tts(self):
        instance = TTS(voice="AWS.Polly.Danielle-Neural")
        try:
            yield instance
        finally:
            await instance.close()

    async def test_stream_audio_yields_chunks(self, tts):
        out = []
        async for item in tts.send_iter(
            "This is a test of the Telnyx text to speech API."
        ):
            out.append(item)

        assert len(out) > 0
        assert out[0].data
        assert out[-1].final

    async def test_decoded_audio_is_audible_and_long_enough(self, tts):
        chunks = [
            item.data
            async for item in tts.send_iter(
                "The quick brown fox jumps over the lazy dog, and then keeps "
                "running for a good while longer across the field."
            )
            if item.data is not None
        ]

        assert chunks
        rates = {chunk.sample_rate for chunk in chunks}
        assert len(rates) == 1
        total_samples = sum(len(chunk.samples) for chunk in chunks)
        # A sentence this long spans more than one MP3 file, so a broken ID3
        # strip truncates the audio well before this threshold.
        assert total_samples / rates.pop() > 3.0
        assert max(abs(int(chunk.samples.max())) for chunk in chunks) > 0

    async def test_second_synthesis_reconnects(self, tts):
        first = [item async for item in tts.send_iter("First utterance.")]
        second = [item async for item in tts.send_iter("Second utterance.")]

        assert any(item.data is not None for item in first)
        assert any(item.data is not None for item in second)
