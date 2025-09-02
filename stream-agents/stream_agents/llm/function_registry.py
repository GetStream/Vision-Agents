"""
Function Registry for LLM Function Calling

This module provides automatic function registration and schema inference
for LLM function calling across all providers.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Dict, List, Callable, Any, Optional, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class FunctionDefinition:
    """Definition of a function that can be called by an LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable
    is_async: bool
    return_type: Optional[type] = None


class FunctionRegistry:
    """Registry for LLM-callable functions with automatic schema inference."""
    
    def __init__(self):
        self._functions: Dict[str, FunctionDefinition] = {}
        self.logger = logging.getLogger("FunctionRegistry")
    
    def function(self, description: str = "", name: Optional[str] = None):
        """
        Decorator to register a function for LLM calling.
        
        Args:
            description: Human-readable description of what the function does
            name: Optional custom name for the function (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            # Infer schema from function signature
            parameters = self._infer_schema(func)
            
            # Get return type hints
            return_type = None
            if hasattr(func, '__annotations__'):
                return_type = func.__annotations__.get('return')
            
            self._functions[func_name] = FunctionDefinition(
                name=func_name,
                description=description or func.__doc__ or f"Call the {func_name} function",
                parameters=parameters,
                handler=func,
                is_async=inspect.iscoroutinefunction(func),
                return_type=return_type
            )
            
            self.logger.info(f"Registered function: {func_name}")
            return func
        
        return decorator
    
    def _infer_schema(self, func: Callable) -> Dict[str, Any]:
        """Infer JSON schema from function signature and type hints."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, param.annotation)
            param_schema = self._type_to_json_schema(param_type)
            
            # Add description from docstring if available
            if func.__doc__:
                # Try to extract parameter description from docstring
                doc_desc = self._extract_param_description(func.__doc__, param_name)
                if doc_desc:
                    param_schema["description"] = doc_desc
            
            properties[param_name] = param_schema
            
            # Add to required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if required:
            schema["required"] = required
            
        return schema
    
    def _type_to_json_schema(self, type_hint: Any) -> Dict[str, Any]:
        """Convert Python type hint to JSON schema."""
        if type_hint == inspect.Parameter.empty or type_hint is None:
            return {"type": "string"}  # Default fallback
        
        # Handle direct types
        if type_hint == str:
            return {"type": "string"}
        elif type_hint == int:
            return {"type": "integer"}
        elif type_hint == float:
            return {"type": "number"}
        elif type_hint == bool:
            return {"type": "boolean"}
        elif type_hint == list:
            return {"type": "array"}
        elif type_hint == dict:
            return {"type": "object"}
        
        # Handle Union types (including Optional)
        origin = get_origin(type_hint)
        if origin is Union:
            args = get_args(type_hint)
            # Handle Optional[T] which is Union[T, None]
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = self._type_to_json_schema(non_none_type)
                return schema
            else:
                # Multiple non-None types - use first one as primary
                return self._type_to_json_schema(args[0])
        
        # Handle List[T]
        elif origin is list:
            args = get_args(type_hint)
            if args:
                item_schema = self._type_to_json_schema(args[0])
                return {"type": "array", "items": item_schema}
            else:
                return {"type": "array"}
        
        # Handle Dict[K, V]
        elif origin is dict:
            args = get_args(type_hint)
            if len(args) >= 2:
                value_schema = self._type_to_json_schema(args[1])
                return {"type": "object", "additionalProperties": value_schema}
            else:
                return {"type": "object"}
        
        # Handle Enum types
        elif isinstance(type_hint, type) and issubclass(type_hint, Enum):
            enum_values = [e.value for e in type_hint]
            return {
                "type": "string",
                "enum": enum_values,
                "description": f"One of: {', '.join(map(str, enum_values))}"
            }
        
        # Handle custom classes (convert to object with properties)
        elif isinstance(type_hint, type) and hasattr(type_hint, '__annotations__'):
            properties = {}
            for field_name, field_type in type_hint.__annotations__.items():
                properties[field_name] = self._type_to_json_schema(field_type)
            
            return {
                "type": "object",
                "properties": properties,
                "description": f"Object of type {type_hint.__name__}"
            }
        
        # Fallback for unknown types
        else:
            self.logger.warning(f"Unknown type hint: {type_hint}, defaulting to string")
            return {"type": "string"}
    
    def _extract_param_description(self, docstring: str, param_name: str) -> Optional[str]:
        """Extract parameter description from docstring."""
        if not docstring:
            return None
        
        lines = docstring.strip().split('\n')
        in_args_section = False
        
        for line in lines:
            line = line.strip()
            
            # Look for Args: or Parameters: section
            if line.lower().startswith(('args:', 'parameters:', 'param ')):
                in_args_section = True
                continue
            
            # Stop at next section
            if in_args_section and line and not line.startswith(' '):
                break
            
            # Look for parameter description
            if in_args_section and line.startswith(param_name):
                # Extract description after parameter name
                parts = line.split(':', 1)
                if len(parts) > 1:
                    return parts[1].strip()
        
        return None
    
    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert registered functions to OpenAI tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": func.name,
                    "description": func.description,
                    "parameters": func.parameters
                }
            }
            for func in self._functions.values()
        ]
    
    def get_gemini_tools(self) -> List[Dict[str, Any]]:
        """Convert registered functions to Gemini tools format."""
        if not self._functions:
            return []
        
        function_declarations = []
        for func in self._functions.values():
            function_declarations.append({
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters
            })
        
        return [{"function_declarations": function_declarations}]
    
    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Convert registered functions to Anthropic tools format."""
        return [
            {
                "name": func.name,
                "description": func.description,
                "input_schema": func.parameters
            }
            for func in self._functions.values()
        ]
    
    async def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered function with given arguments."""
        if name not in self._functions:
            raise ValueError(f"Function '{name}' not registered")
        
        func_def = self._functions[name]
        
        try:
            # Validate arguments against schema
            self._validate_arguments(func_def.parameters, arguments)
            
            # Call the function
            if func_def.is_async:
                result = await func_def.handler(**arguments)
            else:
                result = func_def.handler(**arguments)
            
            self.logger.debug(f"Function {name} executed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing function {name}: {e}")
            raise
    
    def _validate_arguments(self, schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
        """Basic validation of arguments against schema."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in arguments:
                raise ValueError(f"Required parameter '{param}' missing")
        
        # Basic type validation
        for param_name, value in arguments.items():
            if param_name not in properties:
                self.logger.warning(f"Unknown parameter '{param_name}' provided")
                continue
            
            param_schema = properties[param_name]
            expected_type = param_schema.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                raise ValueError(f"Parameter '{param_name}' must be a string")
            elif expected_type == "integer" and not isinstance(value, int):
                raise ValueError(f"Parameter '{param_name}' must be an integer")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                raise ValueError(f"Parameter '{param_name}' must be a number")
            elif expected_type == "boolean" and not isinstance(value, bool):
                raise ValueError(f"Parameter '{param_name}' must be a boolean")
            elif expected_type == "array" and not isinstance(value, list):
                raise ValueError(f"Parameter '{param_name}' must be an array")
            elif expected_type == "object" and not isinstance(value, dict):
                raise ValueError(f"Parameter '{param_name}' must be an object")
    
    def list_functions(self) -> List[str]:
        """Get list of registered function names."""
        return list(self._functions.keys())
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered function."""
        if name not in self._functions:
            return None
        
        func = self._functions[name]
        return {
            "name": func.name,
            "description": func.description,
            "parameters": func.parameters,
            "is_async": func.is_async,
            "return_type": str(func.return_type) if func.return_type else None
        }
