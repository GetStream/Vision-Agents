"""
Test function calling functionality in Gemini Realtime class.
"""

import asyncio
import os
import pytest
from typing import List, Dict, Any
from dotenv import load_dotenv

from vision_agents.plugins import gemini
from vision_agents.core.llm.events import (
    RealtimeResponseEvent,
    RealtimeAudioOutputEvent,
)

# Load environment variables
load_dotenv()


class TestGeminiRealtimeFunctionCalling:
    """Integration tests for function calling in Gemini Realtime class."""

    @pytest.fixture
    async def realtime_instance(self):
        """Create a realtime instance with real Gemini client."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip(
                "GOOGLE_API_KEY or GEMINI_API_KEY not set - skipping integration test"
            )

        # Check if google.genai is available
        try:
            import google.genai  # noqa: F401
        except ImportError as e:
            pytest.skip(f"Required Google packages not available: {e}")

        realtime = gemini.Realtime(
            model="gemini-2.5-flash-native-audio-preview-09-2025", api_key=api_key
        )

        try:
            yield realtime
        finally:
            await realtime.close()

    async def test_convert_tools_to_provider_format(self):
        """Test tool conversion to Gemini Live format."""
        # Create a minimal instance just for testing the conversion method
        realtime = gemini.Realtime(model="test-model", api_key="test-key")

        # Test tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression",
                        }
                    },
                    "required": ["expression"],
                },
            },
        ]

        result = realtime._convert_tools_to_provider_format(tools)

        assert len(result) == 1
        assert "function_declarations" in result[0]
        assert len(result[0]["function_declarations"]) == 2

        # Check first tool
        tool1 = result[0]["function_declarations"][0]
        assert tool1["name"] == "get_weather"
        assert tool1["description"] == "Get weather information"
        assert "location" in tool1["parameters"]["properties"]

        # Check second tool
        tool2 = result[0]["function_declarations"][1]
        assert tool2["name"] == "calculate"
        assert tool2["description"] == "Perform calculations"
        assert "expression" in tool2["parameters"]["properties"]

    @pytest.mark.integration
    async def test_live_function_calling_basic(self, realtime_instance):
        """Test basic live function calling with weather function."""
        realtime = realtime_instance

        # Track function calls and responses
        function_calls: List[Dict[str, Any]] = []
        text_responses: List[str] = []

        # Register a weather function
        @realtime.register_function(description="Get current weather for a location")
        def get_weather(location: str) -> Dict[str, str]:
            """Get weather information for a location."""
            function_calls.append({"name": "get_weather", "location": location})
            return {
                "location": location,
                "temperature": "22°C",
                "condition": "Sunny",
                "humidity": "65%",
            }

        # Set up event listeners for audio output
        @realtime.events.subscribe
        async def handle_audio_output(event: RealtimeAudioOutputEvent):
            if event.data:
                # Audio was received - this indicates Gemini responded
                text_responses.append("audio_response_received")

        @realtime.events.subscribe
        async def handle_response(event: RealtimeResponseEvent):
            if event.text:
                text_responses.append(event.text)

        # Connect and send a prompt that should trigger the function
        await realtime.connect()

        # Send a prompt that encourages function calling
        prompt = "What's the weather like in New York? Please use the get_weather function to check."
        await realtime.simple_response(prompt)

        # Wait for response and function call
        await asyncio.sleep(8.0)

        # Verify function was called
        assert len(function_calls) > 0, "Function was not called by Gemini"
        assert function_calls[0]["name"] == "get_weather"
        assert function_calls[0]["location"] == "New York"

        # Remove the text response assertion

    @pytest.mark.integration
    async def test_live_function_calling_error_handling(self, realtime_instance):
        """Test live function calling with error handling."""
        realtime = realtime_instance

        # Track function calls and responses
        function_calls: List[Dict[str, Any]] = []
        text_responses: List[str] = []

        # Register a function that will raise an error
        @realtime.register_function(description="A function that sometimes fails")
        def unreliable_function(input_data: str) -> Dict[str, Any]:
            """A function that raises an error for testing."""
            function_calls.append({"name": "unreliable_function", "input": input_data})
            if "error" in input_data.lower():
                raise ValueError("Simulated error for testing")
            return {"result": f"Success: {input_data}"}

        # Set up event listeners for audio output
        @realtime.events.subscribe
        async def handle_audio_output(event: RealtimeAudioOutputEvent):
            if event.data:
                # Audio was received - this indicates Gemini responded
                text_responses.append("audio_response_received")

        @realtime.events.subscribe
        async def handle_response(event: RealtimeResponseEvent):
            if event.text:
                text_responses.append(event.text)

        # Connect and send a prompt that should trigger the function with error
        await realtime.connect()

        # Send a prompt that should cause an error
        prompt = "Please call the unreliable_function with 'error test' as input."
        await realtime.simple_response(prompt)

        # Wait for response and function call
        await asyncio.sleep(8.0)

        # Verify function was called
        assert len(function_calls) > 0, "Function was not called by Gemini"
        assert function_calls[0]["name"] == "unreliable_function"

        # Verify we got a response (should mention the error)
        assert len(text_responses) > 0, "No response received from Gemini"

    @pytest.mark.integration
    async def test_live_function_calling_multiple_functions(self, realtime_instance):
        """Test live function calling with multiple functions in one request."""
        realtime = realtime_instance

        # Track function calls
        function_calls: List[Dict[str, Any]] = []
        text_responses: List[str] = []

        # Register multiple functions
        @realtime.register_function(description="Get current time")
        def get_time() -> Dict[str, str]:
            """Get current time."""
            function_calls.append({"name": "get_time"})
            return {"time": "2024-01-15 14:30:00", "timezone": "UTC"}

        @realtime.register_function(description="Get system status")
        def get_status() -> Dict[str, str]:
            """Get system status."""
            function_calls.append({"name": "get_status"})
            return {"status": "healthy", "uptime": "24h"}

        # Set up event listeners for audio output
        @realtime.events.subscribe
        async def handle_audio_output(event: RealtimeAudioOutputEvent):
            if event.data:
                # Audio was received - this indicates Gemini responded
                text_responses.append("audio_response_received")

        @realtime.events.subscribe
        async def handle_response(event: RealtimeResponseEvent):
            if event.text:
                text_responses.append(event.text)

        # Connect and send a prompt that should trigger multiple functions
        await realtime.connect()

        # Send a prompt that encourages multiple function calls
        prompt = "Please check the current time and system status using the available functions."
        await realtime.simple_response(prompt)

        # Wait for response and function calls
        await asyncio.sleep(10.0)

        # Verify functions were called
        assert len(function_calls) >= 2, (
            f"Expected at least 2 function calls, got {len(function_calls)}"
        )

        function_names = [call["name"] for call in function_calls]
        assert "get_time" in function_names, "get_time function was not called"
        assert "get_status" in function_names, "get_status function was not called"

        # Verify we got a response
        assert len(text_responses) > 0, "No response received from Gemini"

    async def test_get_config_with_tools(self):
        """Test that tools are added to the config."""
        # Create a minimal instance for testing config creation
        realtime = gemini.Realtime(model="test-model", api_key="test-key")

        # Register a test function
        @realtime.register_function(description="Test function")
        def test_func(param: str) -> str:
            return f"test: {param}"

        config = realtime.get_config()

        # Verify tools were added
        assert "tools" in config
        assert len(config["tools"]) == 1
        assert "function_declarations" in config["tools"][0]
        assert len(config["tools"][0]["function_declarations"]) == 1
        assert config["tools"][0]["function_declarations"][0]["name"] == "test_func"

    async def test_get_config_without_tools(self):
        """Test config creation when no tools are available."""
        # Create a minimal instance without registering any functions
        realtime = gemini.Realtime(model="test-model", api_key="test-key")

        config = realtime.get_config()

        # Verify tools were not added
        assert "tools" not in config
