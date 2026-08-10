"""Tests for the Telnyx TTS plugin."""

import os
from types import SimpleNamespace

import aiohttp
import av
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


@pytest.fixture
def tts() -> TTS:
    return TTS(api_key="KEY_test")


@pytest.fixture
def dropped_socket_tts(request) -> TTS:
    """A TTS whose socket is dropped mid-send, as stop_audio() would.

    Parametrised with ``stop_before_drop``: True models a concurrent
    ``stop_audio()`` (barge-in), False a genuine connection failure.
    """
    stop_before_drop = request.param
    instance = TTS(api_key="KEY_test")

    class DroppedWS:
        closed = True

        async def send_str(self, data: str) -> None:
            if stop_before_drop:
                instance._stop_event.set()
            raise aiohttp.ClientConnectionResetError("Cannot write to closing")

        async def close(self) -> None:
            return None

    class FakeSession:
        closed = False

        async def ws_connect(self, url: str, headers: dict[str, str]):
            return DroppedWS()

    instance._session = FakeSession()
    return instance


@pytest.fixture
def ws_serving():
    """Build a websocket stand-in that replays ``payloads`` then closes."""

    def build(payloads: list[str]) -> object:
        messages = [
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=payload)
            for payload in payloads
        ] + [SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)]

        class FakeWS:
            def __init__(self) -> None:
                self._queue = list(messages)

            async def receive(self):
                return self._queue.pop(0)

        return FakeWS()

    return build


class TestTelnyxTTS:
    """Unit tests for Telnyx TTS configuration and the receive loop."""

    async def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            TTS()

    async def test_default_configuration(self, tts):
        assert tts.voice == "Telnyx.KokoroTTS.af_heart"
        assert tts.provider_name == "telnyx"

    async def test_custom_voice(self):
        tts = TTS(api_key="KEY_test", voice="AWS.Polly.Danielle-Neural")
        assert tts.voice == "AWS.Polly.Danielle-Neural"

    async def test_voice_is_reported_as_the_model(self):
        """Agent metadata reads ``model`` off the component."""
        tts = TTS(api_key="KEY_test", voice="AWS.Polly.Danielle-Neural")
        assert tts.model == "AWS.Polly.Danielle-Neural"

    @pytest.mark.parametrize("dropped_socket_tts", [True], indirect=True)
    async def test_stop_during_synthesis_ends_quietly(self, dropped_socket_tts):
        """A socket closed by a concurrent stop_audio() is a barge-in."""
        stream = await dropped_socket_tts.stream_audio("hello")
        assert [chunk async for chunk in stream] == []

    @pytest.mark.parametrize("dropped_socket_tts", [False], indirect=True)
    async def test_connection_drop_without_stop_propagates(self, dropped_socket_tts):
        """A stale stop must not silence a genuine failure in a new synthesis."""
        await dropped_socket_tts.stop_audio()

        stream = await dropped_socket_tts.stream_audio("hello")
        with pytest.raises(aiohttp.ClientConnectionError):
            [chunk async for chunk in stream]

    async def test_non_dict_payload_is_skipped(self, tts, ws_serving):
        ws = ws_serving(['["not", "a", "dict"]'])

        assert [chunk async for chunk in tts._receive_audio(ws)] == []

    async def test_invalid_base64_audio_is_skipped(self, tts, ws_serving):
        ws = ws_serving(['{"audio": "!!!not base64!!!"}'])

        assert [chunk async for chunk in tts._receive_audio(ws)] == []

    async def test_non_string_audio_is_skipped(self, tts, ws_serving):
        ws = ws_serving(['{"audio": 12345}'])

        assert [chunk async for chunk in tts._receive_audio(ws)] == []

    async def test_undecodable_audio_is_dropped(self, tts):
        class FailingDecoder:
            def parse(self, data: bytes):
                raise av.InvalidDataError(1094995529, "Invalid data")

        assert (
            tts._decode(b"\xff\xf3junk", FailingDecoder(), None, _Id3Stripper()) == []
        )


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
