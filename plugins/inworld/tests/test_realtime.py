import asyncio

import pytest
from dotenv import load_dotenv
from vision_agents.core.llm.realtime import RealtimeAgentTranscript, RealtimeAudioOutput
from vision_agents.plugins import inworld
from vision_agents.plugins.inworld.tool_utils import convert_tools_to_openai_format

load_dotenv()


class TestInworldRealtime:
    """Unit tests for Inworld Realtime construction, event dispatch, and tool schema."""

    @pytest.fixture
    async def realtime(self):
        rt = inworld.Realtime(api_key="test-key-unit")
        try:
            yield rt
        finally:
            await rt.close()

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

    async def test_agent_transcript_flows(self, realtime):
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

        transcripts = [
            i for i in realtime.output.peek() if isinstance(i, RealtimeAgentTranscript)
        ]
        assert any(t.text == "hello there" for t in transcripts)

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


@pytest.mark.integration
class TestInworldRealtimeIntegration:
    """End-to-end tests against the live Inworld Realtime API.

    Requires ``INWORLD_API_KEY`` in the environment.
    """

    @pytest.fixture
    async def realtime(self):
        rt = inworld.Realtime()
        try:
            await rt.connect()
            yield rt
        finally:
            await rt.close()

    async def test_connect_marks_connected(self, realtime):
        assert realtime.connected is True

    async def test_close_is_idempotent(self, realtime):
        await realtime.close()
        await realtime.close()

    async def test_data_channel_opens_after_connect(self, realtime):
        """End-to-end ICE+DTLS handshake must succeed so the data channel
        opens — otherwise events can't flow and the session is useless.

        This guards against the Inworld-media-behind-NAT class of bug: without
        TURN credentials from /v1/realtime/ice-servers, ICE stalls and this
        test hangs past the timeout.
        """
        for _ in range(150):  # up to 15 s
            if realtime.rtc._data_channel_open_event.is_set():
                break
            await asyncio.sleep(0.1)
        assert realtime.rtc._data_channel_open_event.is_set(), (
            "Data channel did not open within 15 s — check TURN/ICE servers"
        )

    async def test_simple_response_yields_audio_and_transcript(self, realtime):
        """A text prompt should produce audio output and an agent transcript."""
        async for _ in realtime.simple_response("Say hi in one short sentence."):
            pass

        await asyncio.sleep(8.0)

        items = realtime.output.peek()
        audio = [i for i in items if isinstance(i, RealtimeAudioOutput)]
        transcripts = [i for i in items if isinstance(i, RealtimeAgentTranscript)]
        assert len(audio) > 0, "Expected audio output"
        assert any(t.text for t in transcripts), "Expected agent transcript"
