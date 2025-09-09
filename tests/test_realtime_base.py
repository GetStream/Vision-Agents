import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from pyee.asyncio import AsyncIOEventEmitter

from stream_agents.core.agents import Agent
from stream_agents.core.llm import realtime as base_rt
from stream_agents.core.llm.realtime import RealtimeResponse


class FakeConversation:
    def __init__(self) -> None:
        self.partial_calls: list[tuple[str, Optional[Any]]] = []
        self.finish_calls: list[str] = []

    def partial_update_message(self, text: str, participant: Any = None) -> None:
        self.partial_calls.append((text, participant))

    def finish_last_message(self, text: str) -> None:
        self.finish_calls.append(text)


class FakeRealtime(base_rt.Realtime):
    def __init__(self) -> None:
        super().__init__(provider_name="FakeRT")
        # Mark ready immediately
        self._is_connected = True
        self._ready_event.set()

    async def connect(self):
        return None

    async def send_text(self, text: str):
        # Emit transcript for user
        self._emit_transcript_event(text=text, is_user=True)
        # Emit a delta and a final response
        self._emit_response_event(text="Hello", is_complete=False)
        self._emit_response_event(text="Hello world", is_complete=True)

    def send_audio_pcm(self, pcm, target_rate: int = 48000):
        return None

    async def _close_impl(self):
        return None


class _DummyConn(AsyncIOEventEmitter):
    def __init__(self) -> None:
        super().__init__()
        self._ws_client = None

    async def add_tracks(self, audio=None, video=None):
        return None

    async def wait(self):
        # Return immediately
        await asyncio.sleep(0)

    # Decorator-style event registration is provided by AsyncIOEventEmitter


class _ConnCM:
    def __init__(self):
        self.conn = _DummyConn()

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_agent_conversation_updates_with_realtime(monkeypatch):
    # Patch rtc.join to our dummy context manager
    from stream_agents.core.agents import agents as agents_mod

    async def _fake_join(call, user_id, subscription_config=None):
        return _ConnCM()

    monkeypatch.setattr(agents_mod.rtc, "join", _fake_join)

    # Patch StreamConversation constructor used in Agent.join
    fake_conv = FakeConversation()

    class _FakeSC:
        def __init__(self, *a, **k) -> None:
            pass

    monkeypatch.setattr(agents_mod, "StreamConversation", lambda *a, **k: fake_conv)

    # Build Agent with FakeRealtime
    rt = FakeRealtime()
    agent = Agent(llm=rt)

    # Fake Call with minimal attributes used
    call = SimpleNamespace(id="c1", client=SimpleNamespace(stream=SimpleNamespace(chat=SimpleNamespace(get_or_create_channel=lambda *a, **k: SimpleNamespace(data=SimpleNamespace(channel="ch"))))))

    # Run join (which registers event mirroring to conversation)
    mgr = await agent.join(call)

    # Trigger a send_text to produce transcript and response events
    await rt.send_text("Hi")
    # Allow async event handlers to run
    await asyncio.sleep(0.01)

    # Validate conversation updates: partial and final
    assert ("Hello", None) in fake_conv.partial_calls
    assert "Hello world" in fake_conv.finish_calls


class FakeRealtimeAgg(base_rt.Realtime):
    def __init__(self) -> None:
        super().__init__(provider_name="FakeRTAgg")
        self._is_connected = True
        self._ready_event.set()

    async def connect(self):
        return None

    async def send_text(self, text: str):
        # Emit two deltas and a final with small punctuation
        self._emit_response_event(text="Hi ", is_complete=False)
        self._emit_response_event(text="there", is_complete=False)
        self._emit_response_event(text="!", is_complete=True)

    def send_audio_pcm(self, pcm, target_rate: int = 48000):
        return None

    async def _close_impl(self):
        return None


@pytest.mark.asyncio
async def test_simple_response_aggregates_and_returns_realtimeresponse():
    rt = FakeRealtimeAgg()

    # Capture before/after callbacks
    seen_before = {}
    seen_after: list[RealtimeResponse] = []

    def _before(msgs):
        seen_before["count"] = len(msgs)

    async def _after(resp: RealtimeResponse):
        seen_after.append(resp)

    rt.set_before_response_listener(_before)
    rt.set_after_response_listener(_after)

    result = await rt.simple_response(text="start")

    assert isinstance(result, RealtimeResponse)
    assert result.text == "Hi there!"
    assert seen_before.get("count") == 1
    assert len(seen_after) == 1 and seen_after[0].text == "Hi there!"


@pytest.mark.asyncio
async def test_wait_until_ready_returns_true_immediately():
    rt = FakeRealtime()
    assert await rt.wait_until_ready(timeout=0.01) is True


@pytest.mark.asyncio
async def test_close_emits_disconnected_event():
    rt = FakeRealtime()
    observed = {"disconnected": False}

    @rt.on("disconnected")  # type: ignore[arg-type]
    async def _on_disc(_):
        observed["disconnected"] = True

    await rt.close()
    # Allow async event handlers to run
    await asyncio.sleep(0)
    assert observed["disconnected"] is True


@pytest.mark.asyncio
async def test_noop_video_and_playback_methods_do_not_error():
    rt = FakeRealtime()
    # Default base implementations should be safe no-ops
    await rt.start_video_sender(track=None)
    await rt.stop_video_sender()
    await rt.interrupt_playback()
    rt.resume_playback()


class FakeRealtimeNative(base_rt.Realtime):
    def __init__(self) -> None:
        super().__init__(provider_name="FakeRTNative")
        self._is_connected = True
        self._ready_event.set()

    async def connect(self):
        return None

    async def send_text(self, text: str):
        # Not used in native_response test
        pass

    async def native_send_realtime_input(self, *, text=None, audio=None, media=None) -> None:
        # Emit two deltas and an empty final (hybrid contract)
        self._emit_response_event(text="foo", is_complete=False)
        self._emit_response_event(text="bar", is_complete=False)
        self._emit_response_event(text="", is_complete=True)

    def send_audio_pcm(self, pcm, target_rate: int = 48000):
        return None

    async def _close_impl(self):
        return None


@pytest.mark.asyncio
async def test_native_response_aggregates_and_returns_realtimeresponse():
    rt = FakeRealtimeNative()

    result = await rt.native_response(text="x")
    assert isinstance(result, RealtimeResponse)
    assert result.text == "foobar"
