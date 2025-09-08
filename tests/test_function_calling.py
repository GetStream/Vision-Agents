"""
Tests for function calling functionality.
"""

import pytest
from unittest.mock import Mock, patch

from stream_agents.core.llm import FunctionRegistry, function_registry


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
        assert x_param.type == int
        
        y_param = next(p for p in func_def.parameters if p.name == "y")
        assert y_param.required is False
        assert y_param.default == "default"
        assert y_param.type == str
    
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
