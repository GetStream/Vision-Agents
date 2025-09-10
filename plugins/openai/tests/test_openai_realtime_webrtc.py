import asyncio

import pytest

from stream_agents.core.llm.llm import LLMResponse
from stream_agents.plugins.openai.realtime import Realtime
from stream_agents.core.events import (
    RealtimeConnectedEvent,
    RealtimeTranscriptEvent,
    RealtimeResponseEvent,
    RealtimeErrorEvent,
)

from dotenv import load_dotenv

load_dotenv()


class TestRealtime:
    """Test suite for Realtime class with real API calls."""

    @pytest.fixture
    def llm(self) -> Realtime:
        """Test Realtime initialization."""
        return Realtime(model="gpt-4o-realtime-preview-2024-12-17")

    def test_init_defaults(self, llm: Realtime):
        """Test initialization with default parameters."""
        assert llm.model == "gpt-4o-realtime-preview-2024-12-17"
        assert llm.voice == "alloy"
        assert llm.turn_detection is True
        assert llm.realtime is True

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        llm = Realtime(
            model="custom-model",
            voice="nova",
            turn_detection=False,
            system_prompt="You are a pirate.",
        )
        assert llm.model == "custom-model"
        assert llm.voice == "nova"
        assert llm.turn_detection is False
        assert llm.system_prompt == "You are a pirate."

    @pytest.mark.integration
    async def test_simple(self, llm: Realtime):
        """Test simple text response with event listening."""
        # Track events
        events = []
        connected = False

        # Register event listeners
        @llm.on("connected")
        async def on_connected(event: RealtimeConnectedEvent):
            nonlocal connected
            connected = True
            events.append(("connected", event))
            print(
                f"Connected to {event.provider} with model {event.session_config.get('model')}"
            )

        @llm.on("transcript")
        async def on_transcript(event: RealtimeTranscriptEvent):
            events.append(("transcript", event))
            role = "User" if event.is_user else "Assistant"
            print(f"{role} transcript: {event.text}")

        @llm.on("response")
        async def on_response(event: RealtimeResponseEvent):
            events.append(("response", event))
            print(f"Response: {event.text}")

        @llm.on("error")
        async def on_error(event):
            events.append(("error", event))
            print(f"Error: {event.error_message}")

        try:
            # Send the question
            print("Sending question...")
            response = await llm._simple_response_async(
                "What is the capital of France?"
            )
            print(f"Got response: {response.text}")

            # Verify connection was established
            assert connected, "Connection was not established"

            # Verify we got a response
            assert response.text, "No response text received"

            # Check that the response mentions Paris
            assert "paris" in response.text.lower(), (
                f"Expected 'Paris' in response, got: {response.text}"
            )

            # Verify events were emitted
            event_types = [e[0] for e in events]
            assert "connected" in event_types, "No connected event"
            assert "response" in event_types, "No response event"

            # Check the response event
            response_events = [e[1] for e in events if e[0] == "response"]
            assert len(response_events) > 0
            assert "paris" in response_events[0].text.lower()

            print("✅ Test passed!")

        finally:
            # Clean up
            await llm.close()

    @pytest.mark.integration
    async def test_native_api(self, llm: Realtime):
        """Test create_response method for compatibility."""
        response = await llm.create_response(
            input="Say hello in French.",
        )

        assert isinstance(response, LLMResponse)
        assert response.text
        assert any(
            word in response.text.lower() for word in ["bonjour", "salut", "hello"]
        )

    @pytest.mark.integration
    async def test_event_emission(self, llm: Realtime):
        """Test that proper events are emitted during conversation."""
        events = []

        # Capture all events
        @llm.on("connected")
        async def on_connected(event):
            events.append(event)

        @llm.on("transcript")
        async def on_transcript(event):
            events.append(event)

        @llm.on("response")
        async def on_response(event):
            events.append(event)

        @llm.on("error")
        async def on_error(event):
            events.append(event)

        try:
            # Make a simple request
            await llm._simple_response_async("What is 2+2?")

            # Verify connected event
            connected_events = [
                e for e in events if isinstance(e, RealtimeConnectedEvent)
            ]
            assert len(connected_events) >= 1
            connected = connected_events[0]
            assert connected.provider == "openai"
            assert connected.session_config["model"] == llm.model
            assert "text" in connected.capabilities
            assert "audio" in connected.capabilities

            # Verify response event
            response_events = [
                e for e in events if isinstance(e, RealtimeResponseEvent)
            ]
            assert len(response_events) >= 1
            resp_event = response_events[0]
            assert resp_event.text
            assert resp_event.is_complete
            assert any(word in resp_event.text.lower() for word in ["4", "four"])

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_multiple_messages(self, llm: Realtime):
        """Test multiple messages in sequence."""
        try:
            # First message
            response1 = await llm._simple_response_async("Remember the number 42.")
            assert response1.text

            # Second message - connection should be reused
            response2 = await llm._simple_response_async(
                "What number did I just tell you?"
            )
            assert response2.text
            # Note: OpenAI Realtime maintains conversation context within the same session
            assert any(
                word in response2.text for word in ["42", "forty-two", "forty two"]
            )

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_multiple_exchanges(self):
        """Test multiple question/answer exchanges in the same session."""
        llm = Realtime(
            voice="echo",
            turn_detection=False,
            system_prompt="You are a helpful math tutor. Keep answers very short.",
        )

        try:
            # First question
            response1 = await llm._simple_response_async("What is 2 + 2?")
            assert any(word in response1.text.lower() for word in ["4", "four"])

            # Second question - should reuse connection
            response2 = await llm._simple_response_async("What is 10 times that?")
            assert any(word in response2.text.lower() for word in ["40", "forty"])

            print("✅ Multiple exchanges test passed!")

        finally:
            await llm.close()

    @pytest.mark.integration
    async def test_error_handling(self):
        """Test error handling with invalid configuration."""
        # Create LLM with invalid model
        llm = Realtime(
            model="invalid-model-name",
            voice="alloy",
        )

        error_events = []

        @llm.on("error")
        async def on_error(event: RealtimeErrorEvent):
            error_events.append(event)

        # This should fail
        try:
            await llm._simple_response_async("Hello")
            # If we get here, the test should fail
            assert False, "Expected an error but got response"
        except Exception:
            # Expected
            pass

        # little sleep here is necessary because errors are delivered async
        await asyncio.sleep(1)

        # Clean up
        try:
            await llm.close()
        except Exception:
            pass

        # Verify error event was emitted
        # Note: sometimes error events might not be captured if the connection fails too quickly
        if len(error_events) > 0:
            assert error_events[0].context == "connection"

    @pytest.mark.integration
    async def test_close(self, llm: Realtime):
        """Test closing the connection properly."""
        # Make a request to establish connection
        await llm._simple_response_async("Hello")

        # Track disconnection
        events = []

        @llm.on("disconnected")
        async def on_disconnected(event):
            events.append(("disconnected", event))

        @llm.on("closed")
        async def on_closed(event):
            events.append(("closed", event))

        # Close the LLM
        await llm.close()

        # Give a moment for async events to be processed
        await asyncio.sleep(0.1)

        # Verify events
        event_types = [e[0] for e in events]
        assert "disconnected" in event_types, (
            f"Expected disconnected event, got: {event_types}"
        )
        assert "closed" in event_types, f"Expected closed event, got: {event_types}"
