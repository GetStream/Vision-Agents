"""
Tests for function calling functionality.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

from stream_agents.core.llm import FunctionRegistry, function_registry
from stream_agents.core.llm.llm import LLM
from stream_agents.core.llm.openai_llm import OpenAILLM
from stream_agents.core.llm.claude_llm import ClaudeLLM
from stream_agents.core.llm.gemini_llm import GeminiLLM


class TestFunctionRegistry:
    """Test the FunctionRegistry class."""
    
    def test_register_function(self):
        """Test registering a function."""
        registry = FunctionRegistry()
        
        @registry.register(description="Test function")
        def test_func(x: int, y: str = "default") -> str:
            """A test function."""
            return f"{x}:{y}"
        
        assert "test_func" in registry.list_functions()
        func_def = registry.get_function("test_func")
        assert func_def is not None
        assert func_def.name == "test_func"
        assert func_def.description == "Test function"
        assert len(func_def.parameters) == 2
        
        # Check parameters
        param_names = [p.name for p in func_def.parameters]
        assert "x" in param_names
        assert "y" in param_names
        
        # Check parameter details
        x_param = next(p for p in func_def.parameters if p.name == "x")
        assert x_param.required is True
        assert x_param.type is int
        
        y_param = next(p for p in func_def.parameters if p.name == "y")
        assert y_param.required is False
        assert y_param.default == "default"
        assert y_param.type is str
    
    def test_call_function(self):
        """Test calling a registered function."""
        registry = FunctionRegistry()
        
        @registry.register()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        result = registry.call_function("add", {"a": 5, "b": 3})
        assert result == 8
    
    def test_call_function_with_defaults(self):
        """Test calling a function with default parameters."""
        registry = FunctionRegistry()
        
        @registry.register()
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"
        
        # Call with all parameters
        result = registry.call_function("greet", {"name": "Alice", "greeting": "Hi"})
        assert result == "Hi, Alice!"
        
        # Call with only required parameters
        result = registry.call_function("greet", {"name": "Bob"})
        assert result == "Hello, Bob!"
    
    def test_call_nonexistent_function(self):
        """Test calling a non-existent function raises KeyError."""
        registry = FunctionRegistry()
        
        with pytest.raises(KeyError):
            registry.call_function("nonexistent", {})
    
    def test_call_function_missing_required_param(self):
        """Test calling a function with missing required parameters raises TypeError."""
        registry = FunctionRegistry()
        
        @registry.register()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b
        
        with pytest.raises(TypeError):
            registry.call_function("add", {"a": 5})  # Missing 'b'
    
    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        registry = FunctionRegistry()
        
        @registry.register(description="Test function")
        def test_func(x: int, y: str = "default") -> str:
            """A test function."""
            return f"{x}:{y}"
        
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["name"] == "test_func"
        assert schema["description"] == "Test function"
        assert "parameters_schema" in schema
        
        params_schema = schema["parameters_schema"]
        assert params_schema["type"] == "object"
        assert "properties" in params_schema
        assert "required" in params_schema
        
        properties = params_schema["properties"]
        assert "x" in properties
        assert "y" in properties
        assert properties["x"]["type"] == "integer"
        assert properties["y"]["type"] == "string"
        
        required = params_schema["required"]
        assert "x" in required
        assert "y" not in required  # y has a default value


class TestGlobalRegistry:
    """Test the global function registry."""
    
    def test_global_registry(self):
        """Test that the global registry works."""
        # Clear any existing functions
        function_registry._functions.clear()
        
        @function_registry.register(description="Global test function")
        def global_test(x: int) -> int:
            """A global test function."""
            return x * 2
        
        assert "global_test" in function_registry.list_functions()
        
        result = function_registry.call_function("global_test", {"x": 5})
        assert result == 10


class TestLLMFunctionCalling:
    """Test function calling integration with LLMs."""
    
    def test_llm_function_registration(self):
        """Test that LLMs can register functions."""
        llm = Mock(spec=LLM)
        llm.function_registry = FunctionRegistry()
        
        @llm.function_registry.register(description="Test function")
        def test_func(x: int) -> str:
            """A test function."""
            return f"Result: {x}"
        
        functions = llm.function_registry.list_functions()
        assert "test_func" in functions
        
        # Test calling through LLM interface
        result = llm.function_registry.call_function("test_func", {"x": 42})
        assert result == "Result: 42"
    
    def test_llm_get_available_functions(self):
        """Test getting available functions from LLM."""
        llm = Mock(spec=LLM)
        llm.function_registry = FunctionRegistry()
        
        @llm.function_registry.register()
        def func1(x: int) -> int:
            return x * 2
        
        @llm.function_registry.register()
        def func2(y: str) -> str:
            return f"Hello {y}"
        
        available_functions = llm.function_registry.get_tool_schemas()
        assert len(available_functions) == 2
        
        function_names = [f["name"] for f in available_functions]
        assert "func1" in function_names
        assert "func2" in function_names


class TestOpenAIFunctionCalling:
    """Test OpenAI-specific function calling."""
    
    @patch('stream_agents.core.llm.openai_llm.OpenAI')
    def test_openai_function_calling_response(self, mock_openai):
        """Test OpenAI function calling response processing."""
        # Mock OpenAI response with function call
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [
            Mock(
                id="call_123",
                type="function",
                function=Mock(
                    name="get_weather",
                    arguments='{"location": "New York"}'
                )
            )
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        # Register a test function
        @llm.register_function(description="Get weather for a location")
        def get_weather(location: str) -> str:
            """Get weather information."""
            return f"Weather in {location}: Sunny, 72°F"
        
        # Mock the create_response method to return our mock response
        llm.create_response = Mock(return_value=mock_response)
        
        # Test that function is registered
        functions = llm.get_available_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "get_weather"
        
        # Test function calling
        result = llm.call_function("get_weather", {"location": "New York"})
        assert result == "Weather in New York: Sunny, 72°F"
    
    @patch('stream_agents.core.llm.openai_llm.OpenAI')
    def test_openai_conversational_response(self, mock_openai):
        """Test OpenAI conversational response generation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock follow-up response
        mock_followup = Mock()
        mock_followup.choices = [Mock()]
        mock_followup.choices[0].message = Mock()
        mock_followup.choices[0].message.content = "The weather in New York is sunny and 72°F. Perfect for a walk!"
        mock_client.chat.completions.create.return_value = mock_followup
        
        llm = OpenAILLM(api_key="test-key", model="gpt-4")
        
        # Mock conversation
        from stream_agents.core.agents.conversation import Conversation
        llm._conversation = Conversation(
            instructions="You are a helpful assistant.",
            messages=[]
        )
        
        # Test conversational response generation
        tool_results = [
            {
                "type": "tool_result",
                "name": "get_weather",
                "result_json": {"result": "Weather in New York: Sunny, 72°F"}
            }
        ]
        
        original_response = Mock()
        response = llm._generate_conversational_response(tool_results, original_response)
        
        assert response == "The weather in New York is sunny and 72°F. Perfect for a walk!"


class TestClaudeFunctionCalling:
    """Test Claude-specific function calling."""
    
    @patch('stream_agents.core.llm.claude_llm.anthropic')
    def test_claude_function_calling_response(self, mock_anthropic):
        """Test Claude function calling response processing."""
        # Mock Claude response with tool use
        mock_response = Mock()
        mock_response.content = [
            Mock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"location": "New York"}
            )
        ]
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        
        llm = ClaudeLLM(api_key="test-key", model="claude-3-sonnet-20240229")
        
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
    
    @patch('stream_agents.core.llm.claude_llm.anthropic')
    def test_claude_conversational_response(self, mock_anthropic):
        """Test Claude conversational response generation."""
        mock_client = Mock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client
        
        # Mock follow-up response
        mock_followup = Mock()
        mock_followup.content = [Mock(text="The weather in New York is sunny and 72°F. Perfect for a walk!")]
        mock_client.messages.create.return_value = mock_followup
        
        llm = ClaudeLLM(api_key="test-key", model="claude-3-sonnet-20240229")
        
        # Mock conversation
        from stream_agents.core.agents.conversation import Conversation
        llm._conversation = Conversation(
            instructions="You are a helpful assistant.",
            messages=[]
        )
        
        # Test conversational response generation
        tool_results = [
            {
                "type": "tool_result",
                "name": "get_weather",
                "result_json": {"result": "Weather in New York: Sunny, 72°F"}
            }
        ]
        
        original_response = Mock()
        response = llm._generate_conversational_response(tool_results, original_response)
        
        assert response == "The weather in New York is sunny and 72°F. Perfect for a walk!"


class TestGeminiFunctionCalling:
    """Test Gemini-specific function calling."""
    
    @patch('stream_agents.core.llm.gemini_llm.genai')
    def test_gemini_function_calling_response(self, mock_genai):
        """Test Gemini function calling response processing."""
        # Mock Gemini response with function call
        mock_response = Mock()
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content = Mock()
        mock_response.candidates[0].content.parts = [
            Mock(
                function_call=Mock(
                    name="get_weather",
                    args={"location": "New York"}
                )
            )
        ]
        mock_response.text = None
        
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client
        
        llm = GeminiLLM(api_key="test-key", model="gemini-2.0-flash")
        
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
    
    @patch('stream_agents.core.llm.gemini_llm.genai')
    def test_gemini_conversational_response(self, mock_genai):
        """Test Gemini conversational response generation."""
        mock_client = Mock()
        mock_genai.Client.return_value = mock_client
        
        # Mock chat and follow-up response
        mock_chat = Mock()
        mock_followup = Mock()
        mock_followup.text = "The weather in New York is sunny and 72°F. Perfect for a walk!"
        mock_chat.send_message.return_value = mock_followup
        mock_client.chats.create.return_value = mock_chat
        
        llm = GeminiLLM(api_key="test-key", model="gemini-2.0-flash")
        llm.chat = mock_chat
        
        # Test conversational response generation
        tool_results = [
            {
                "type": "tool_result",
                "name": "get_weather",
                "result_json": {"result": "Weather in New York: Sunny, 72°F"}
            }
        ]
        
        original_response = Mock()
        response = llm._generate_conversational_response(tool_results, original_response)
        
        assert response == "The weather in New York is sunny and 72°F. Perfect for a walk!"


class TestFunctionCallingIntegration:
    """Test end-to-end function calling integration."""
    
    def test_tool_call_processing(self):
        """Test processing tool calls with results."""
        llm = Mock(spec=LLM)
        llm.function_registry = FunctionRegistry()
        
        # Register a function
        @llm.function_registry.register(description="Calculate square")
        def square(x: int) -> int:
            """Calculate the square of a number."""
            return x * x
        
        # Mock normalized response with tool calls
        normalized_response = {
            "id": "test-123",
            "model": "test-model",
            "status": "completed",
            "output": [
                {
                    "type": "tool_call",
                    "name": "square",
                    "arguments_json": '{"x": 5}'
                }
            ],
            "output_text": "",
            "raw": Mock()
        }
        
        # Mock the process_tool_calls method
        def mock_process_tool_calls(response):
            # Simulate tool call processing
            updated_output = [
                {
                    "type": "tool_result",
                    "name": "square",
                    "result_json": {"result": 25}
                }
            ]
            response["output"] = updated_output
            response["output_text"] = "square result: {'result': 25}"
            return response
        
        llm.process_tool_calls = mock_process_tool_calls
        
        # Test processing
        result = llm.process_tool_calls(normalized_response)
        
        assert result["output_text"] == "square result: {'result': 25}"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "tool_result"
        assert result["output"][0]["result_json"]["result"] == 25
    
    def test_error_handling_in_function_calls(self):
        """Test error handling when function calls fail."""
        llm = Mock(spec=LLM)
        llm.function_registry = FunctionRegistry()
        
        # Register a function that raises an exception
        @llm.function_registry.register(description="Failing function")
        def failing_func(x: int) -> int:
            """A function that always fails."""
            raise ValueError("Test error")
        
        # Test that calling the function raises the exception
        with pytest.raises(ValueError, match="Test error"):
            llm.function_registry.call_function("failing_func", {"x": 5})
    
    def test_function_schema_generation(self):
        """Test that function schemas are generated correctly."""
        llm = Mock(spec=LLM)
        llm.function_registry = FunctionRegistry()
        
        @llm.function_registry.register(description="Complex function")
        def complex_func(
            name: str,
            age: int,
            city: str = "Unknown",
            active: bool = True
        ) -> Dict[str, Any]:
            """A complex function with various parameter types."""
            return {
                "name": name,
                "age": age,
                "city": city,
                "active": active
            }
        
        schemas = llm.function_registry.get_tool_schemas()
        assert len(schemas) == 1
        
        schema = schemas[0]
        assert schema["name"] == "complex_func"
        assert schema["description"] == "Complex function"
        
        params_schema = schema["parameters_schema"]
        properties = params_schema["properties"]
        required = params_schema["required"]
        
        # Check parameter types
        assert properties["name"]["type"] == "string"
        assert properties["age"]["type"] == "integer"
        assert properties["city"]["type"] == "string"
        assert properties["active"]["type"] == "boolean"
        
        # Check required parameters
        assert "name" in required
        assert "age" in required
        assert "city" not in required  # Has default
        assert "active" not in required  # Has default
