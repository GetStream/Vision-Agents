"""
Function registry for managing available functions that can be called by LLMs.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from dataclasses import dataclass
from enum import Enum

from .llm_types import ToolSchema


@dataclass
class FunctionParameter:
    """Represents a parameter of a function."""

    name: str
    type: Type
    description: Optional[str] = None
    required: bool = True
    default: Any = None


@dataclass
class FunctionDefinition:
    """Represents a complete function definition."""

    name: str
    description: str
    parameters: List[FunctionParameter]
    function: Callable
    returns: Optional[Type] = None


@dataclass
class _ExplicitSchemaFunction:
    """Represents a function with an explicitly provided JSON schema (e.g., MCP tools)."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    function: Callable


class FunctionRegistry:
    """Registry for managing available functions that can be called by LLMs."""

    def __init__(self):
        self._functions: Dict[str, FunctionDefinition | _ExplicitSchemaFunction] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
    ) -> Callable:
        """
        Decorator to register a function with the registry.

        Args:
            name: Optional custom name for the function. If not provided, uses the function name.
            description: Optional description for the function. If not provided, uses the docstring.
            parameters_schema: Optional explicit JSON schema for parameters. If provided, this is
                used instead of inferring from type hints. Useful for MCP tools or external APIs.

        Returns:
            Decorator function.
        """

        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            func_description = description or func.__doc__ or ""

            # If explicit schema provided, use it directly
            if parameters_schema is not None:
                self._functions[func_name] = _ExplicitSchemaFunction(
                    name=func_name,
                    description=func_description,
                    parameters_schema=parameters_schema,
                    function=func,
                )
                return func

            # Otherwise, extract from type hints
            type_hints = get_type_hints(func)
            sig = inspect.signature(func)

            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = type_hints.get(param_name, type(None))
                param_description = None

                # Check if there's a docstring with parameter descriptions
                if func.__doc__:
                    # Simple extraction of parameter descriptions from docstring
                    # This is a basic implementation - could be enhanced with proper parsing
                    pass

                parameters.append(
                    FunctionParameter(
                        name=param_name,
                        type=param_type,
                        description=param_description,
                        required=param.default == inspect.Parameter.empty,
                        default=param.default
                        if param.default != inspect.Parameter.empty
                        else None,
                    )
                )

            # Determine return type
            return_type = type_hints.get("return", None)

            function_def = FunctionDefinition(
                name=func_name,
                description=func_description,
                parameters=parameters,
                function=func,
                returns=return_type,
            )

            self._functions[func_name] = function_def
            return func

        return decorator

    def get_function(
        self, name: str
    ) -> Optional[FunctionDefinition | _ExplicitSchemaFunction]:
        """Get a function definition by name."""
        return self._functions.get(name)

    def list_functions(self) -> List[str]:
        """Get a list of all registered function names."""
        return list(self._functions.keys())

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Get tool schemas for all registered functions."""
        schemas = []
        for func_def in self._functions.values():
            if isinstance(func_def, _ExplicitSchemaFunction):
                # Use the explicitly provided schema
                schemas.append(
                    ToolSchema(
                        name=func_def.name,
                        description=func_def.description,
                        parameters_schema=func_def.parameters_schema,
                    )
                )
            else:
                schema = self._function_to_tool_schema(func_def)
                schemas.append(schema)
        return schemas

    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a registered function with the given arguments.

        Args:
            name: Name of the function to call.
            arguments: Dictionary of arguments to pass to the function.

        Returns:
            Result of the function call.

        Raises:
            KeyError: If the function is not registered.
            TypeError: If the arguments don't match the function signature.
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' is not registered")

        func_def = self._functions[name]

        # For explicit schema functions, just call directly (validation is external)
        if isinstance(func_def, _ExplicitSchemaFunction):
            return func_def.function(**arguments)

        # Validate required parameters for introspected functions
        for param in func_def.parameters:
            if param.required and param.name not in arguments:
                raise TypeError(
                    f"Missing required parameter '{param.name}' for function '{name}'"
                )

        # Call the function with the provided arguments
        return func_def.function(**arguments)

    def get_callable(self, name: str) -> Callable:
        """
        Get the callable function by name.

        Args:
            name: Name of the function

        Returns:
            The callable function

        Raises:
            KeyError: If the function is not registered
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' is not registered")

        return self._functions[name].function

    def _function_to_tool_schema(self, func_def: FunctionDefinition) -> ToolSchema:
        """Convert a function definition to a tool schema."""
        properties = {}
        required = []

        for param in func_def.parameters:
            param_schema = self._type_to_json_schema(param.type)
            if param.description:
                param_schema["description"] = param.description

            properties[param.name] = param_schema

            if param.required:
                required.append(param.name)

        schema = {"type": "object", "properties": properties}

        if required:
            schema["required"] = required

        return ToolSchema(
            name=func_def.name,
            description=func_def.description,
            parameters_schema=schema,
        )

    def _type_to_json_schema(self, type_hint: Type) -> Dict[str, Any]:
        """Convert a Python type hint to a JSON schema."""
        # Handle basic types
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is list or type_hint is List:
            return {"type": "array"}
        elif type_hint is dict or type_hint is Dict:
            return {"type": "object"}

        # Handle Optional types
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
            # Check if it's Optional (Union[SomeType, None])
            args = type_hint.__args__
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                return self._type_to_json_schema(non_none_type)

        # Handle List types
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is list:
            if hasattr(type_hint, "__args__") and type_hint.__args__:
                item_type = type_hint.__args__[0]
                return {"type": "array", "items": self._type_to_json_schema(item_type)}
            return {"type": "array"}

        # Handle Dict types
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is dict:
            return {"type": "object"}

        # Handle Enum types
        if inspect.isclass(type_hint) and issubclass(type_hint, Enum):
            return {"type": "string", "enum": [e.value for e in type_hint]}

        # Default fallback
        return {"type": "string"}


# Global registry instance
function_registry = FunctionRegistry()
