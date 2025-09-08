import asyncio
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from pyee.asyncio import AsyncIOEventEmitter

from stream_agents.core.agents import Agent
from stream_agents.core.llm import realtime as base_rt
from stream_agents.core.events import STSResponseEvent, STSTranscriptEvent


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

