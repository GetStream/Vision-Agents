import asyncio
import importlib
import sys
import types
from typing import Any, Dict, List, cast
from unittest.mock import AsyncMock

import numpy as np
import pytest

from getstream.plugins.gemini.live import GeminiLive
from getstream.video.rtc.track_util import PcmData


class _FakeSession:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def send_realtime_input(self, *, media=None, audio=None, text=None):
        self.calls.append({"media": media, "audio": audio, "text": text})


class _FakeImage:
    def __init__(self, arr) -> None:
        self._arr = arr

    def save(self, fp, format="PNG"):
        # Write a minimal PNG header to simulate an encoded image
        fp.write(b"\x89PNG\r\n\x1a\n")


class _FakeImageModule:
    @staticmethod
    def fromarray(arr):
        return _FakeImage(arr)


class _FakeVideoTrack:
    kind = "video"

    def __init__(self, frames: int = 1) -> None:
        self._remaining = frames

    async def recv(self):
        if self._remaining <= 0:
            await asyncio.sleep(0)  # yield control
        self._remaining -= 1

        # Return an object that supports to_ndarray("rgb24")
        class _Frame:
            @staticmethod
            def to_ndarray(format: str = "rgb24"):
                return (np.random.rand(16, 16, 3) * 255).astype(np.uint8)

        return _Frame()


# --- Pytest fixtures ---
@pytest.fixture
def fake_image(monkeypatch):
    import getstream.plugins.gemini.live.live as live_mod

    monkeypatch.setattr(live_mod, "Image", _FakeImageModule)
    return _FakeImageModule


@pytest.mark.asyncio
async def test_start_video_sender_sends_media_blob(fake_image):
    g = GeminiLive(api_key="test", model="test-model")
    # Pretend we're connected and have a session
    fake = _FakeSession()
    g._session = fake  # type: ignore[attr-defined]
    g._is_connected = True  # type: ignore[attr-defined]

    track = _FakeVideoTrack(frames=2)
    await g.start_video_sender(track, fps=100)  # type: ignore[reportArgumentType]
    # Allow the loop to run at least once
    await asyncio.sleep(0.05)
    await g.stop_video_sender()

    # We should have at least one media send
    assert any(call.get("media") is not None for call in fake.calls)


@pytest.mark.asyncio
async def test_send_audio_pcm_sends_audio_blob():
    g = GeminiLive(api_key="test", model="test-model")
    fake = _FakeSession()
    g._session = fake  # type: ignore[attr-defined]
    g._is_connected = True  # type: ignore[attr-defined]

    samples = (np.zeros(1600)).astype(np.int16)
    pcm = PcmData(samples=samples, sample_rate=16000, format="s16")
    await g.send_audio_pcm(pcm, target_rate=16000)

    # Ensure an audio blob was sent
    assert any(call.get("audio") is not None for call in fake.calls)


@pytest.mark.asyncio
async def test_stop_video_sender_cancels_task(fake_image):
    g = GeminiLive(api_key="test", model="test-model")
    g._session = _FakeSession()  # type: ignore[attr-defined]
    g._is_connected = True  # type: ignore[attr-defined]

    track = _FakeVideoTrack(frames=1)
    await g.start_video_sender(track, fps=100)  # type: ignore[reportArgumentType]
    await asyncio.sleep(0)
    await g.stop_video_sender()

    # Task should be cleared
    assert g._video_sender_task is None  # type: ignore[attr-defined]


# --- Provide mocked google.genai SDK so the module under test can import ---
class _DummyBlob:
    def __init__(self, data=None, mime_type=None):
        self.data = data
        self.mime_type = mime_type


class _DummyDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _DummyModality:
    AUDIO = "audio"


class _DummyTurnCoverage:
    TURN_INCLUDES_ONLY_ACTIVITY = "turn_only"


_types_mod: Any = types.ModuleType("google.genai.types")
_types_mod.LiveConnectConfigDict = _DummyDict
_types_mod.Blob = _DummyBlob
_types_mod.Modality = _DummyModality
_types_mod.AudioTranscriptionConfigDict = _DummyDict
_types_mod.RealtimeInputConfigDict = _DummyDict
_types_mod.TurnCoverage = _DummyTurnCoverage
_types_mod.SpeechConfigDict = _DummyDict
_types_mod.VoiceConfigDict = _DummyDict
_types_mod.PrebuiltVoiceConfigDict = _DummyDict
_types_mod.ContextWindowCompressionConfigDict = _DummyDict
_types_mod.SlidingWindowDict = _DummyDict

_live_mod: Any = types.ModuleType("google.genai.live")
_live_mod.AsyncSession = object


# A controllable dummy session/context for tests
class _DummySession:
    def __init__(self, responses=None):
        # list of (data_bytes, text_str) tuples
        self._responses = responses or []
        self._receive_calls = 0
        self.sent = []

    async def send_realtime_input(self, text=None, audio=None):
        self.sent.append({"text": text, "audio": audio})

    def receive(self):
        async def _gen():
            # yield responses only on first call to receive(); then be empty
            if self._receive_calls == 0:
                for data, text in self._responses:
                    obj = types.SimpleNamespace()
                    obj.data = data
                    setattr(obj, "text", text)
                    yield obj
            self._receive_calls += 1
            return

        return _gen()


class _DummyConnectCM:
    def __init__(self, responses=None):
        self._responses = responses or []
        self.session = None

    async def __aenter__(self):
        self.session = _DummySession(self._responses)
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummyLive:
    def __init__(self, responses=None):
        self._responses = responses or []

    def connect(self, model=None, config=None):
        return _DummyConnectCM(self._responses)


class _DummyClient:
    def __init__(self, *args, **kwargs):
        # tests will mutate this instance's live.responses via monkeypatch
        self.aio = types.SimpleNamespace(live=_DummyLive([]))


_genai_mod: Any = types.ModuleType("google.genai")
_genai_mod.Client = _DummyClient

# Ensure root google package exists for environments that lack it
try:
    importlib.import_module("google")
except ImportError:  # pragma: no cover - environment should have google from protobuf
    pkg: Any = types.ModuleType("google")
    pkg.__path__ = []  # mark as package
    sys.modules["google"] = pkg


from getstream.plugins.gemini.live import live as gemini_live  # noqa: E402


@pytest.fixture
def patch_genai_client(monkeypatch):
    def _patch(responses=None):
        client = _DummyClient()
        client.aio.live = _DummyLive(responses or [])
        monkeypatch.setattr(gemini_live, "Client", lambda *a, **k: client)
        return client

    return _patch


@pytest.fixture
def spy_track_write(monkeypatch):
    def _spy(sts):
        write_spy = AsyncMock()
        monkeypatch.setattr(sts.output_track, "write", write_spy)
        return write_spy

    return _spy


@pytest.fixture
def spy_track_flush(monkeypatch):
    def _spy(sts):
        flush_spy = AsyncMock()
        monkeypatch.setattr(sts.output_track, "flush", flush_spy)
        return flush_spy

    return _spy


@pytest.mark.asyncio
async def test_connect_emits_events_and_forwards_audio_and_text(
    patch_genai_client, spy_track_write
):
    # Arrange streamed responses
    responses = [(b"abc", "hello"), (b"def", None)]
    patch_genai_client(responses)

    events = {"connected": False, "audio": [], "text": []}

    sts = GeminiLive(api_key="key", model="model", provider_config=None)

    @sts.on("connected")  # type: ignore[arg-type]
    async def _on_connected():
        events["connected"] = True

    @sts.on("audio")  # type: ignore[arg-type]
    async def _on_audio(data: bytes):
        events["audio"].append(data)

    @sts.on("text")  # type: ignore[arg-type]
    async def _on_text(text: str):
        events["text"].append(text)

    # Spy on track writes
    write_spy = spy_track_write(sts)

    # Act
    ready = await sts.wait_until_ready(timeout=1.0)
    assert ready is True

    # Give the background receiver loop time to process
    await asyncio.sleep(0.05)

    # Assert
    assert events["connected"] is True
    assert events["audio"] == [b"abc", b"def"]
    assert events["text"] == ["hello"]
    # Track writes should mirror audio chunks while playback is enabled
    write_spy.assert_any_await(b"abc")
    write_spy.assert_any_await(b"def")

    await sts.close()


@pytest.mark.asyncio
async def test_send_text_calls_underlying_session(patch_genai_client):
    patch_genai_client([])

    sts = GeminiLive(api_key="key", model="model", provider_config=None)
    await sts.wait_until_ready(timeout=1.0)

    # Access dummy session to inspect sent calls
    session = cast(Any, sts._session)
    assert session is not None

    await sts.send_text("hi there")

    assert any(call.get("text") == "hi there" for call in session.sent)


@pytest.mark.asyncio
async def test_send_audio_pcm_resample_barge_in_and_silence_timeout(
    monkeypatch, patch_genai_client, spy_track_flush
):
    # No incoming responses needed for this test
    patch_genai_client([])

    # Mock the imported components at the module level
    monkeypatch.setattr(gemini_live, "Blob", _DummyBlob)
    monkeypatch.setattr(gemini_live, "resample_audio", lambda arr, src, dst: arr)

    sts = GeminiLive(
        api_key="key",
        model="model",
        provider_config=None,
        activity_threshold=10,
        silence_timeout_ms=20,
        barge_in=True,
    )
    await sts.wait_until_ready(timeout=1.0)

    # Threshold and timeout configured via constructor

    # Spy on flush
    flush_spy = spy_track_flush(sts)

    # Build loud mono PCM ndarray at 16k to trigger resample path
    samples = np.ones(480, dtype=np.int16) * 1000  # 30ms at 16k
    pcm = PcmData(format="s16", sample_rate=16000, samples=samples)

    # Act
    await sts.send_audio_pcm(pcm, target_rate=48000)

    # Barge-in should have interrupted playback and flushed
    assert sts._playback_enabled is False
    flush_spy.assert_awaited()

    # Underlying session should have received a blob with correct mime
    session = cast(Any, sts._session)
    assert session is not None and session.sent
    sent_audio = next(call["audio"] for call in session.sent if call.get("audio"))
    assert isinstance(sent_audio, _DummyBlob)
    assert sent_audio.mime_type == "audio/pcm;rate=48000"
    assert isinstance(sent_audio.data, (bytes, bytearray))

    # Wait for silence timeout to re-enable playback
    await asyncio.sleep(0.05)
    assert sts._user_speaking is False
    assert sts._playback_enabled is True

    await sts.close()


@pytest.mark.asyncio
async def test_interrupt_and_resume_playback(patch_genai_client, spy_track_flush):
    patch_genai_client([])

    sts = GeminiLive(api_key="key", model="model", provider_config=None)
    await sts.wait_until_ready(timeout=1.0)

    flush_spy = spy_track_flush(sts)

    await sts.interrupt_playback()
    assert sts._playback_enabled is False
    flush_spy.assert_awaited()

    sts.resume_playback()
    assert sts._playback_enabled is True

    await sts.close()


@pytest.mark.asyncio
async def test_stop_response_listener_cancels_task(patch_genai_client):
    # Start with one response so the listener spins up
    responses = [(b"x", None)]
    patch_genai_client(responses)

    sts = GeminiLive(api_key="key", model="model", provider_config=None)
    await sts.wait_until_ready(timeout=1.0)

    # Ensure listener task exists
    await asyncio.sleep(0.02)
    assert sts._audio_receiver_task is not None

    await sts.stop_response_listener()
    assert sts._audio_receiver_task is None

    await sts.close()
