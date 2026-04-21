import asyncio

import pytest
from dotenv import load_dotenv
from vision_agents.core.llm.events import (
    RealtimeAgentSpeechTranscriptionEvent,
    RealtimeConnectedEvent,
)
from vision_agents.plugins import inworld
from vision_agents.plugins.inworld.tool_utils import convert_tools_to_openai_format

load_dotenv()


class TestRealtime:
    """Inworld Realtime plugin tests.

    Unit tests exercise in-process logic (construction, event dispatch,
    tool schema). Integration tests (marked `@pytest.mark.integration`)
    require ``INWORLD_API_KEY`` in the environment.
    """

    @pytest.fixture
    async def realtime(self):
        rt = inworld.Realtime(api_key="test-key-unit")
        try:
            yield rt
        finally:
            await rt.close()

    # --- Unit ---

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("INWORLD_API_KEY", raising=False)
        with pytest.raises(ValueError, match="INWORLD_API_KEY"):
            inworld.Realtime()

    async def test_default_model_and_voice(self, realtime):
        assert realtime.model == "openai/gpt-4o-mini"
        assert realtime.voice == "Dennis"
        assert realtime.realtime_session["model"] == "openai/gpt-4o-mini"
        assert realtime.realtime_session["audio"]["output"]["voice"] == "Dennis"

    async def test_custom_model_and_voice(self):
        rt = inworld.Realtime(
            api_key="test-key",
            model="google-ai-studio/gemini-2.5-flash",
            voice="Olivia",
        )
        try:
            assert rt.model == "google-ai-studio/gemini-2.5-flash"
            assert rt.voice == "Olivia"
        finally:
            await rt.close()

    async def test_instructions_propagate_to_session(self):
        rt = inworld.Realtime(api_key="test-key", instructions="be concise")
        try:
            assert rt.realtime_session["instructions"] == "be concise"
        finally:
            await rt.close()

    async def test_set_instructions_updates_realtime_session(self, realtime):
        realtime.set_instructions("speak like a pirate")
        assert realtime.realtime_session["instructions"] == "speak like a pirate"

    async def test_tool_registration_appears_in_session_config(self, realtime):
        @realtime.register_function()
        async def get_weather(city: str) -> str:
            """Return the weather for a city."""
            return f"sunny in {city}"

        tools = convert_tools_to_openai_format(
            realtime.get_available_functions(), for_realtime=True
        )
        names = [t["name"] for t in tools]
        assert "get_weather" in names

    async def test_interrupt_increments_epoch(self, realtime):
        before = realtime.epoch
        await realtime.interrupt()
        assert realtime.epoch == before + 1

    async def test_agent_transcript_event_flows(self, realtime):
        received: list[RealtimeAgentSpeechTranscriptionEvent] = []

        @realtime.events.subscribe
        async def _on(event: RealtimeAgentSpeechTranscriptionEvent):
            received.append(event)

        await realtime._handle_inworld_event(
            {
                "type": "response.output_audio_transcript.done",
                "event_id": "evt_1",
                "response_id": "resp_1",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": 0,
                "transcript": "hello there",
            }
        )
        # Allow the event bus to drain
        await asyncio.sleep(0.05)
        assert any(e.text == "hello there" for e in received)

    async def test_unknown_event_type_is_swallowed(self, realtime):
        # Should not raise
        await realtime._handle_inworld_event(
            {"type": "some.unknown.event", "payload": {}}
        )

    async def test_inworld_specific_schema_drift_does_not_crash(self, realtime):
        """Inworld's events drift from OpenAI's pydantic schema (e.g.
        response.done has metadata.attempts as a list, content types of
        'text'/'audio' instead of 'input_text'/'output_text', role
        'assistant' on some items; input_audio_transcription.completed
        omits 'usage'). The handler must tolerate these without raising."""
        # Real-world shape from the live API
        await realtime._handle_inworld_event(
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "event_id": "cdc3f369-d3",
                "item_id": "item_1",
                "transcript": "Hello, can you hear me?",
            }
        )
        await realtime._handle_inworld_event(
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "metadata": {
                        "attempts": [
                            {"model": "google-vertex", "credential_type": "system"}
                        ]
                    },
                    "output": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "hi"},
                                {"type": "audio"},
                            ],
                        }
                    ],
                },
            }
        )

    # --- Integration (INWORLD_API_KEY required) ---

    @pytest.fixture
    async def live_realtime(self):
        rt = inworld.Realtime()
        try:
            yield rt
        finally:
            await rt.close()

    @pytest.mark.integration
    async def test_connect_emits_connected_event(self, live_realtime):
        received: list[RealtimeConnectedEvent] = []

        @live_realtime.events.subscribe
        async def _on(event: RealtimeConnectedEvent):
            received.append(event)

        await live_realtime.connect()
        await asyncio.sleep(0.2)
        assert received, "Expected at least one RealtimeConnectedEvent"

    @pytest.mark.integration
    async def test_close_is_idempotent(self, live_realtime):
        await live_realtime.connect()
        await live_realtime.close()
        await live_realtime.close()

    @pytest.mark.integration
    async def test_data_channel_opens_after_connect(self, live_realtime):
        """End-to-end ICE+DTLS handshake must succeed so the data channel
        opens — otherwise events can't flow and the session is useless.

        This guards against the Inworld-media-behind-NAT class of bug: without
        TURN credentials from /v1/realtime/ice-servers, ICE stalls and this
        test hangs past the timeout.
        """
        await live_realtime.connect()
        for _ in range(150):  # up to 15 s
            if live_realtime.rtc._data_channel_open_event.is_set():
                break
            await asyncio.sleep(0.1)
        assert live_realtime.rtc._data_channel_open_event.is_set(), (
            "Data channel did not open within 15 s — check TURN/ICE servers"
        )
