"""
Test function calling functionality in Gemini Realtime class.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from stream_agents.plugins.gemini.gemini_realtime import Realtime


class TestGeminiRealtimeFunctionCalling:
    """Test function calling in Gemini Realtime class."""

    @pytest.fixture
    def realtime_instance(self):
        """Create a realtime instance with mocked dependencies."""
        with patch('stream_agents.core.llm.llm.EventManager') as mock_event_manager:
            with patch('google.genai.Client') as mock_client:
                # Mock the event manager to avoid event loop issues
                mock_event_manager.return_value = Mock()
                
                # Mock the Gemini client
                mock_client.return_value = Mock()
                
                realtime = Realtime(model="test-model", api_key="test-key")
                return realtime

    def test_convert_tools_to_provider_format(self, realtime_instance):
        """Test tool conversion to Gemini Live format."""
        realtime = realtime_instance
        
        # Mock tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "calculate",
                "description": "Perform calculations",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"]
                }
            }
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

    def test_extract_tool_calls_from_response(self, realtime_instance):
        """Test extraction of tool calls from Gemini Live response."""
        realtime = realtime_instance
        
        # Mock response with function calls
        mock_part = Mock()
        mock_part.function_call = Mock()
        mock_part.function_call.name = "get_weather"
        mock_part.function_call.args = {"location": "New York"}
        mock_part.function_call.id = "call_123"
        
        mock_model_turn = Mock()
        mock_model_turn.parts = [mock_part]
        
        mock_server_content = Mock()
        mock_server_content.model_turn = mock_model_turn
        
        mock_response = Mock()
        mock_response.server_content = mock_server_content
        
        calls = realtime._extract_tool_calls_from_response(mock_response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments_json"] == {"location": "New York"}
        assert calls[0]["id"] == "call_123"
        assert calls[0]["type"] == "tool_call"

    def test_extract_tool_calls_from_response_no_calls(self, realtime_instance):
        """Test extraction when no function calls are present."""
        realtime = realtime_instance
        
        # Mock response without function calls
        mock_part = Mock()
        mock_part.text = "Hello world"
        del mock_part.function_call  # No function call
        
        mock_model_turn = Mock()
        mock_model_turn.parts = [mock_part]
        
        mock_server_content = Mock()
        mock_server_content.model_turn = mock_model_turn
        
        mock_response = Mock()
        mock_response.server_content = mock_server_content
        
        calls = realtime._extract_tool_calls_from_response(mock_response)
        
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_handle_function_call_success(self, realtime_instance):
        """Test successful function call handling."""
        realtime = realtime_instance
        
        # Mock function call
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "New York"}
        mock_function_call.id = "call_123"
        
        # Mock the tool execution
        with patch.object(realtime, '_run_one_tool', new_callable=AsyncMock) as mock_run_tool:
            mock_run_tool.return_value = (
                {"name": "get_weather", "arguments_json": {"location": "New York"}},
                {"temperature": "22°C", "condition": "Sunny"},
                None
            )
            
            # Mock the response sending
            with patch.object(realtime, '_send_function_response', new_callable=AsyncMock) as mock_send_response:
                await realtime._handle_function_call(mock_function_call)
                
                # Verify tool was executed
                mock_run_tool.assert_called_once()
                
                # Verify response was sent
                mock_send_response.assert_called_once_with(
                    "get_weather",
                    {"temperature": "22°C", "condition": "Sunny"},
                    "call_123"
                )

    @pytest.mark.asyncio
    async def test_handle_function_call_error(self, realtime_instance):
        """Test function call handling with error."""
        realtime = realtime_instance
        
        # Mock function call
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "New York"}
        mock_function_call.id = "call_123"
        
        # Mock the tool execution to return an error
        with patch.object(realtime, '_run_one_tool', new_callable=AsyncMock) as mock_run_tool:
            mock_run_tool.return_value = (
                {"name": "get_weather", "arguments_json": {"location": "New York"}},
                None,
                Exception("API Error")
            )
            
            # Mock the response sending
            with patch.object(realtime, '_send_function_response', new_callable=AsyncMock) as mock_send_response:
                await realtime._handle_function_call(mock_function_call)
                
                # Verify tool was executed
                mock_run_tool.assert_called_once()
                
                # Verify error response was sent
                mock_send_response.assert_called_once_with(
                    "get_weather",
                    {"error": "API Error"},
                    "call_123"
                )

    @pytest.mark.asyncio
    async def test_send_function_response(self, realtime_instance):
        """Test sending function response back to Gemini Live."""
        realtime = realtime_instance
        
        # Mock the session
        mock_session = AsyncMock()
        realtime._session = mock_session
        
        # Mock the types module
        with patch('stream_agents.plugins.gemini.gemini_realtime.types') as mock_types:
            mock_part = Mock()
            mock_types.Part.from_function_response.return_value = mock_part
            
            await realtime._send_function_response(
                "get_weather",
                {"temperature": "22°C"},
                "call_123"
            )
            
            # Verify the function response part was created
            mock_types.Part.from_function_response.assert_called_once_with(
                name="get_weather",
                response={"temperature": "22°C"}
            )
            
            # Verify it was sent to the session
            mock_session.send_realtime_input.assert_called_once_with(parts=[mock_part])

    def test_create_config_with_tools(self, realtime_instance):
        """Test that tools are added to the config."""
        realtime = realtime_instance
        
        # Mock available functions
        with patch.object(realtime, 'get_available_functions') as mock_get_functions:
            mock_get_functions.return_value = [
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters_schema": {"type": "object", "properties": {}}
                }
            ]
            
            config = realtime._create_config()
            
            # Verify tools were added
            assert "tools" in config
            assert len(config["tools"]) == 1
            assert "function_declarations" in config["tools"][0]
            assert len(config["tools"][0]["function_declarations"]) == 1
            assert config["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    def test_create_config_without_tools(self, realtime_instance):
        """Test config creation when no tools are available."""
        realtime = realtime_instance
        
        # Mock no available functions
        with patch.object(realtime, 'get_available_functions') as mock_get_functions:
            mock_get_functions.return_value = []
            
            config = realtime._create_config()
            
            # Verify tools were not added
            assert "tools" not in config
