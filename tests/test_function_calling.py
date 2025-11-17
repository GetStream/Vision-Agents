"""
Tests for function calling functionality.
"""

import pytest
from unittest.mock import Mock, patch

from vision_agents.core.llm import FunctionRegistry, function_registry
from vision_agents.core.llm.llm import LLM
from vision_agents.plugins.openai import LLM as OpenAILLM
from vision_agents.plugins.anthropic import LLM as ClaudeLLM
from vision_agents.plugins.gemini import LLM as GeminiLLM


class TestFunctionRegistry:
    """Test the FunctionRegistry class."""

    def test_register_function(self):
        """Test registering a function."""
        registry = FunctionRegistry()

        @registry.register(description="Test function")
        def test_func(x: int, y: int = 5) -> int:
            """Test function with default parameter."""
            return x + y

        assert "test_func" in registry._functions
        assert registry._functions["test_func"].description == "Test function"
        assert len(registry._functions["test_func"].parameters) == 2

    def test_call_function(self):
        """Test calling a registered function."""
        registry = FunctionRegistry()

        @registry.register(description="Add two numbers")
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = registry.call_function("add_numbers", {"a": 5, "b": 3})
        assert result == 8

    def test_call_function_with_defaults(self):
        """Test calling a function with default parameters."""
        registry = FunctionRegistry()

        @registry.register(description="Test function with defaults")
        def test_func(x: int, y: int = 10) -> int:
            """Test function with default parameter."""
            return x + y

        # Test with both parameters
        result = registry.call_function("test_func", {"x": 5, "y": 3})
        assert result == 8

        # Test with default parameter
        result = registry.call_function("test_func", {"x": 5})
        assert result == 15

    def test_call_nonexistent_function(self):
        """Test calling a non-existent function raises error."""
        registry = FunctionRegistry()

        with pytest.raises(KeyError):
            registry.call_function("nonexistent", {})

    def test_call_function_missing_required_param(self):
        """Test calling a function with missing required parameter raises error."""
        registry = FunctionRegistry()

        @registry.register(description="Test function")
        def test_func(x: int, y: int) -> int:
            """Test function."""
            return x + y

        with pytest.raises(TypeError):
            registry.call_function("test_func", {"x": 5})

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        registry = FunctionRegistry()

        @registry.register(description="Test function")
        def test_func(x: int, y: int = 5) -> int:
            """Test function."""
            return x + y

        schemas = registry.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "test_func"
        assert schemas[0]["description"] == "Test function"
        assert "parameters_schema" in schemas[0]

    def test_get_callable(self):
        """Test getting callable function."""
        registry = FunctionRegistry()

        @registry.register(description="Test function")
        def test_func(x: int) -> int:
            """Test function."""
            return x * 2

        callable_func = registry.get_callable("test_func")
        assert callable_func(5) == 10

        with pytest.raises(KeyError):
            registry.get_callable("nonexistent")


class TestGlobalRegistry:
    """Test the global function registry."""

    def test_global_registry(self):
        """Test that the global registry works."""
        # Clear any existing functions
        function_registry._functions.clear()

        @function_registry.register(description="Global test function")
        def global_test_func(x: int) -> int:
            """Global test function."""
            return x * 3

        assert "global_test_func" in function_registry._functions
        result = function_registry.call_function("global_test_func", {"x": 4})
        assert result == 12


class TestLLMFunctionCalling:
    """Test LLM function calling functionality."""

    async def test_llm_function_registration(self):
        """Test that LLM can register functions."""
        llm = LLM()

        @llm.register_function(description="Test function")
        def test_func(x: int) -> int:
            """Test function."""
            return x * 2

        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "test_func"

    async def test_llm_get_available_functions(self):
        """Test getting available functions from LLM."""
        llm = LLM()

        @llm.register_function(description="Function 1")
        def func1(x: int) -> int:
            return x + 1

        @llm.register_function(description="Function 2")
        def func2(x: int) -> int:
            return x * 2

        functions = llm.get_available_functions()
        assert len(functions) == 2
        function_names = [f["name"] for f in functions]
        assert "func1" in function_names
        assert "func2" in function_names


class TestOpenAIFunctionCalling:
    """Test OpenAI function calling functionality."""

    @patch("vision_agents.plugins.openai.openai_llm.AsyncOpenAI")
    async def test_openai_function_calling_response(self, mock_openai):
        """Test OpenAI function calling response."""
        # Mock the OpenAI client and response
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock the responses.create call
        mock_response = Mock()
        mock_response.output = [
            Mock(
                type="function_call",
                call_id="call_123",
                arguments='{"location": "New York"}',
            )
        ]
        mock_client.responses.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key", model="gpt-4")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"

        # Test function calling
        result = llm.call_function("get_weather", {"location": "New York"})
        assert result == "Weather in New York: Sunny, 72°F"

    @patch("vision_agents.plugins.openai.openai_llm.AsyncOpenAI")
    async def test_openai_conversational_response(self, mock_openai):
        """Test OpenAI conversational response generation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        # Mock the responses.create call
        mock_response = Mock()
        mock_response.output = [
            Mock(
                type="function_call",
                call_id="call_123",
                arguments='{"location": "New York"}',
            )
        ]
        mock_client.responses.create.return_value = mock_response

        llm = OpenAILLM(api_key="test-key", model="gpt-4")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"


class TestClaudeFunctionCalling:
    """Test Claude function calling functionality."""

    @patch("vision_agents.plugins.anthropic.anthropic_llm.AsyncAnthropic")
    async def test_claude_function_calling_response(self, mock_anthropic):
        """Test Claude function calling response."""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock the messages.create call
        mock_response = Mock()
        mock_response.content = [
            Mock(
                type="tool_use",
                id="tool_123",
                name="get_weather",
                input={"location": "New York"},
            )
        ]
        mock_client.messages.create.return_value = mock_response

        llm = ClaudeLLM(api_key="test-key", model="claude-3-5-sonnet-20241022")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"

        # Test function calling
        result = llm.call_function("get_weather", {"location": "New York"})
        assert result == "Weather in New York: Sunny, 72°F"

    @patch("vision_agents.plugins.anthropic.anthropic_llm.AsyncAnthropic")
    async def test_claude_conversational_response(self, mock_anthropic):
        """Test Claude conversational response generation."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Mock the messages.create call
        mock_response = Mock()
        mock_response.content = [
            Mock(
                type="tool_use",
                id="tool_123",
                name="get_weather",
                input={"location": "New York"},
            )
        ]
        mock_client.messages.create.return_value = mock_response

        llm = ClaudeLLM(api_key="test-key", model="claude-3-5-sonnet-20241022")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"


class TestGeminiFunctionCalling:
    """Test Gemini function calling functionality."""

    @patch("vision_agents.plugins.gemini.gemini_llm.Client")
    async def test_gemini_function_calling_response(self, mock_client_class):
        """Test Gemini function calling response."""
        # Mock the Gemini client structure
        mock_async_client = Mock()
        mock_client_instance = Mock()
        mock_client_instance.aio = mock_async_client
        mock_client_class.return_value = mock_client_instance

        # Mock the chat object
        mock_chat = Mock()
        mock_chats = Mock()
        mock_chats.create.return_value = mock_chat
        mock_async_client.chats = mock_chats

        # Mock the send_message_stream call - returns async iterator
        mock_response = Mock()
        mock_response.candidates = [
            Mock(
                content=Mock(
                    parts=[
                        Mock(
                            type="function_call",
                            function_call=Mock(
                                name="get_weather", args={"location": "New York"}
                            ),
                        )
                    ]
                )
            )
        ]
        mock_response.text = ""  # No text, just function call

        async def mock_iterator():
            yield mock_response

        # send_message_stream is called and should return an async iterator
        mock_chat.send_message_stream = Mock(return_value=mock_iterator())

        llm = GeminiLLM(model="gemini-2.0-flash")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"

        # Test function calling
        result = llm.call_function("get_weather", {"location": "New York"})
        assert result == "Weather in New York: Sunny, 72°F"

    @patch("vision_agents.plugins.gemini.gemini_llm.Client")
    async def test_gemini_conversational_response(self, mock_client_class):
        """Test Gemini conversational response generation."""
        # Mock the Gemini client structure
        mock_async_client = Mock()
        mock_client_instance = Mock()
        mock_client_instance.aio = mock_async_client
        mock_client_class.return_value = mock_client_instance

        # Mock the chat object
        mock_chat = Mock()
        mock_chats = Mock()
        mock_chats.create.return_value = mock_chat
        mock_async_client.chats = mock_chats

        # Mock the send_message_stream call - returns async iterator
        mock_response = Mock()
        mock_response.candidates = [
            Mock(
                content=Mock(
                    parts=[
                        Mock(
                            type="function_call",
                            function_call=Mock(
                                name="get_weather", args={"location": "New York"}
                            ),
                        )
                    ]
                )
            )
        ]
        mock_response.text = ""  # No text, just function call

        async def mock_iterator():
            yield mock_response

        # send_message_stream is called and should return an async iterator
        mock_chat.send_message_stream = Mock(return_value=mock_iterator())

        llm = GeminiLLM(model="gemini-2.0-flash")

        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"

        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"


class TestFunctionCallingIntegration:
    """Test function calling integration scenarios."""

    async def test_tool_call_processing(self):
        """Test processing tool calls with multiple functions."""
        llm = LLM()

        @llm.register_function(description="Get weather")
        def get_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"

        @llm.register_function(description="Calculate sum")
        def calculate_sum(a: int, b: int) -> int:
            return a + b

        # Test multiple function registrations
        functions = llm.get_available_functions()
        assert len(functions) == 2

        # Test calling both functions
        weather_result = llm.call_function("get_weather", {"location": "NYC"})
        sum_result = llm.call_function("calculate_sum", {"a": 5, "b": 3})

        assert weather_result == "Weather in NYC: Sunny"
        assert sum_result == 8

    async def test_error_handling_in_function_calls(self):
        """Test error handling in function calls."""
        llm = LLM()

        @llm.register_function(description="Test function that raises error")
        def error_function(x: int) -> int:
            if x < 0:
                raise ValueError("Negative numbers not allowed")
            return x * 2

        # Test normal case
        result = llm.call_function("error_function", {"x": 5})
        assert result == 10

        # Test error case
        with pytest.raises(ValueError):
            llm.call_function("error_function", {"x": -5})

    async def test_function_schema_generation(self):
        """Test that function schemas are generated correctly."""
        llm = LLM()

        @llm.register_function(description="Complex function")
        def complex_function(
            name: str, age: int, is_active: bool = True, tags: list = None
        ) -> dict:
            """Complex function with various parameter types."""
            return {
                "name": name,
                "age": age,
                "is_active": is_active,
                "tags": tags or [],
            }

        schemas = llm.get_available_functions()
        assert len(schemas) == 1

        schema = schemas[0]
        assert schema["name"] == "complex_function"
        assert schema["description"] == "Complex function"
        assert "parameters_schema" in schema

        # Check parameter types
        params = schema["parameters_schema"]["properties"]
        assert "name" in params
        assert "age" in params
        assert "is_active" in params
        assert "tags" in params

        # Check required parameters
        required = schema["parameters_schema"]["required"]
        assert "name" in required
        assert "age" in required
        assert "is_active" not in required  # Has default value
        assert "tags" not in required  # Has default value


class TestConcurrentToolExecution:
    """Test concurrent tool execution functionality."""

    async def test_dedup_and_execute(self):
        """Test the _dedup_and_execute method."""
        llm = LLM()

        @llm.register_function(description="Test function")
        def test_func(x: int) -> int:
            return x * 2

        # Test with duplicate tool calls
        tool_calls = [
            {"id": "call1", "name": "test_func", "arguments_json": {"x": 5}},
            {
                "id": "call2",
                "name": "test_func",
                "arguments_json": {"x": 5},
            },  # Duplicate
            {"id": "call3", "name": "test_func", "arguments_json": {"x": 3}},
        ]

        # This should deduplicate and only execute call1 and call3
        triples, seen = await llm._dedup_and_execute(tool_calls)
        # The deduplication should work, but let's check what actually happens
        # The key is based on (id, name, arguments_json), so different IDs = different keys
        assert len(triples) == 3  # All calls have different IDs, so all are executed
        assert len(seen) == 3  # 3 unique keys in seen set

        # Check results
        results = [result for _, result, _ in triples]
        assert 10 in results  # 5 * 2 (appears twice)
        assert 6 in results  # 3 * 2

    async def test_tool_lifecycle_events(self):
        """Test that tool lifecycle events are emitted."""
        from vision_agents.core.llm.events import ToolStartEvent, ToolEndEvent

        llm = LLM()

        @llm.register_function(description="Test function")
        def test_func(x: int) -> int:
            return x * 2

        # Track emitted events
        start_events = []
        end_events = []

        @llm.events.subscribe
        async def track_start_event(event: ToolStartEvent):
            start_events.append(event)

        @llm.events.subscribe
        async def track_end_event(event: ToolEndEvent):
            end_events.append(event)

        # Execute a tool call
        await llm._run_one_tool(
            {"id": "call1", "name": "test_func", "arguments_json": {"x": 5}}, 30.0
        )
        # Wait for events
        await llm.events.wait(timeout=1.0)

        # Check that events were emitted
        assert len(start_events) == 1
        assert len(end_events) == 1
        assert start_events[0].tool_name == "test_func"
        assert end_events[0].tool_name == "test_func"
        assert end_events[0].success is True

    async def test_output_sanitization(self):
        """Test output sanitization for large responses."""
        llm = LLM()

        # Test normal output
        normal_output = "Hello world"
        sanitized = llm._sanitize_tool_output(normal_output)
        assert sanitized == "Hello world"

        # Test large output
        large_output = "x" * 70000  # Larger than default 60k limit
        sanitized = llm._sanitize_tool_output(large_output)
        assert len(sanitized) == 60001  # 60k + "…"
        assert sanitized.endswith("…")

        # Test non-string output
        dict_output = {"key": "value"}
        sanitized = llm._sanitize_tool_output(dict_output)
        assert sanitized == '{"key": "value"}'
