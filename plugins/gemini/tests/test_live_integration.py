import os
import asyncio

import pytest
from typing import List, TypedDict, Dict, List as TList


class _Events(TypedDict):
    audio: List[bytes]
    text: List[str]


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # If python-dotenv is not installed, ignore and skip the test
    pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_live_with_real_api():
    """
    Optional smoke test: requires GOOGLE_API_KEY and google-genai installed.
    Connects, sends a short text, and asserts we receive audio or text back.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key is None:
        pytest.skip("GOOGLE_API_KEY not set – skipping live Gemini test")

    # Require google-genai to be present
    try:
        import google.genai  # noqa: F401
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"Required Google packages not available: {e}")

    from stream_agents.plugins import gemini

    # Set up instance and event capture

    events: _Events = {"audio": [], "text": []}
    sts = gemini.Realtime(api_key=api_key)

    @sts.on("audio")  # type: ignore[arg-type]
    async def _on_audio(data: bytes):
        events["audio"].append(data)

    @sts.on("text")  # type: ignore[arg-type]
    async def _on_text(text: str):
        events["text"].append(text)

    ready = await sts.wait_until_ready(timeout=10.0)
    if not ready:
        await sts.close()
        raise RuntimeError("Gemini Live did not become ready in time")

    # Send a very short prompt and wait briefly for any response
    await sts.send_text("Speak a short sentence.")

    # Wait up to 10s for any audio or text response
    for _ in range(50):
        if events["audio"] or events["text"]:
            break
        await asyncio.sleep(0.2)

    try:
        assert events["audio"] or events["text"], (
            "No response received from Gemini Live"
        )
    finally:
        await sts.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_live_simple_and_native_and_video():
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key is None:
        pytest.skip("GOOGLE_API_KEY not set – skipping live Gemini test")

    try:
        import google.genai  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Required Google packages not available: {e}")

    from stream_agents.plugins import gemini
    from aiortc.mediastreams import MediaStreamTrack
    import numpy as np

    class _FakeVideo(MediaStreamTrack):
        kind = "video"

        async def recv(self):
            class _F:
                def to_ndarray(self, format="rgb24"):
                    return (np.random.rand(16, 16, 3) * 255).astype(np.uint8)

            return _F()

    events: _Events = {"audio": [], "text": []}
    sts = gemini.Realtime(api_key=api_key)

    @sts.on("audio")  # type: ignore[arg-type]
    async def _on_audio(data: bytes):
        events["audio"].append(data)

    @sts.on("text")  # type: ignore[arg-type]
    async def _on_text(text: str):
        events["text"].append(text)

    # Wait ready
    ready = await sts.wait_until_ready(timeout=10.0)
    if not ready:
        await sts.close()
        raise RuntimeError("Gemini Live did not become ready in time")

    # Send one video frame stream in background
    await sts.start_video_sender(_FakeVideo(), fps=1)

    # Standard simple_response and native paths
    await sts.simple_response(text="Say a short sentence.")
    await sts.native_send_realtime_input(text="Say a second sentence.")

    for _ in range(50):
        if events["audio"] or events["text"]:
            break
        await asyncio.sleep(0.2)

    try:
        assert events["audio"] or events["text"], (
            "No response received from Gemini Live"
        )
    finally:
        await sts.stop_video_sender()
        await sts.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gemini_native_passthrough(monkeypatch):
    # Patch client and session to capture native calls
    from stream_agents.plugins import gemini as gemini_pkg

    calls: Dict[str, TList[object]] = {"text": [], "audio": [], "media": []}

    class _Sess:
        async def send_realtime_input(self, *, text=None, audio=None, media=None):
            if text:
                calls["text"].append(text)
            if audio:
                calls["audio"].append(audio)
            if media:
                calls["media"].append(media)

    class _CM:
        async def __aenter__(self):
            return _Sess()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Live:
        def connect(self, model=None, config=None):
            return _CM()

    class _Client:
        def __init__(self, *a, **k):
            self.aio = type("A", (), {"live": _Live()})

    monkeypatch.setattr(gemini_pkg.realtime, "Client", lambda *a, **k: _Client())

    sts = gemini_pkg.Realtime(api_key="x", model="m")
    ready = await sts.wait_until_ready(timeout=1.0)
    assert ready

    # Call passthrough
    await sts.native_send_realtime_input(text="hello")
    assert calls["text"] == ["hello"]
    await sts.close()


@pytest.mark.asyncio
async def test_gemini_native_response_returns_standardized(monkeypatch):
    import types

    # Patch client to have a dummy session that yields two text parts then ends
    from stream_agents.plugins.gemini import realtime as gemini_live

    class _Sess:
        def __init__(self):
            self._receive_calls = 0

        async def send_realtime_input(self, *, text=None, audio=None, media=None):
            return None

        def receive(self):
            async def _gen():
                if self._receive_calls == 0:
                    for data, text in [(b"x", "Hi "), (b"y", "there")]:
                        obj = types.SimpleNamespace()
                        obj.data = data
                        setattr(obj, "text", text)
                        yield obj
                self._receive_calls += 1
                return

            return _gen()

    class _CM:
        async def __aenter__(self):
            return _Sess()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _Live:
        def connect(self, model=None, config=None):
            return _CM()

    class _Client:
        def __init__(self, *a, **k):
            self.aio = types.SimpleNamespace(live=_Live())

    monkeypatch.setattr(gemini_live, "Client", lambda *a, **k: _Client())

    sts = gemini_live.Realtime(api_key="x", model="m")
    await sts.wait_until_ready(timeout=1.0)

    result = await sts.native_response(text="Say hi")
    from stream_agents.core.llm.realtime import RealtimeResponse

    assert isinstance(result, RealtimeResponse)
    assert result.text == "Hi there"
    await sts.close()
