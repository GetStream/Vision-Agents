"""
Function registry for managing available functions that can be called by LLMs.
"""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_type_hints,
)
from dataclasses import dataclass
from enum import Enum

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError

from .llm_types import ToolSchema

# A number the model sends for a string, such as a zip code, is taken as its digits.
_ARGUMENT_CONFIG = ConfigDict(coerce_numbers_to_str=True)


class ToolError(Exception):
    """A tool call failed in a way the model should hear about.

    The message is returned to the model as the call's error. Any other exception a
    function raises is logged, and the model is only told that the tool failed.
    """


class ToolArgumentsError(ToolError, TypeError):
    """A call's arguments are not what its function takes."""


class ToolCallRefused(ToolError):
    """An ``authorize`` callable refused a call."""


@dataclass
class FunctionParameter:
    """Represents a parameter of a function."""

    name: str
    type: Type
    description: Optional[str] = None
    required: bool = True
    default: Any = None
    adapter: Optional[TypeAdapter] = None


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

    def __init__(
        self,
        authorize: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
    ):
        """
        Args:
            authorize: Optional async callable run before every call with the function's
                name and arguments. Raise ``ToolCallRefused`` to stop the call; its
                message is what the model is told.
        """
        self.functions: Dict[str, FunctionDefinition | _ExplicitSchemaFunction] = {}
        self.authorize = authorize

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
            if not inspect.iscoroutinefunction(func):
                raise ValueError(
                    f"Only async functions can be registered, "
                    f"but '{func.__name__}' is synchronous"
                )

            func_name = name or func.__name__
            func_description = description or func.__doc__ or ""

            # If explicit schema provided, use it directly
            if parameters_schema is not None:
                self.functions[func_name] = _ExplicitSchemaFunction(
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
                # *args and **kwargs are not arguments the model fills in by name.
                if param.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
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
                        adapter=_adapter_for(param_type)
                        if param_name in type_hints
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

            self.functions[func_name] = function_def
            return func

        return decorator

    def get_function(
        self, name: str
    ) -> Optional[FunctionDefinition | _ExplicitSchemaFunction]:
        """Get a function definition by name."""
        return self.functions.get(name)

    def list_functions(self) -> List[str]:
        """Get a list of all registered function names."""
        return list(self.functions.keys())

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Get tool schemas for all registered functions."""
        schemas = []
        for func_def in self.functions.values():
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

    async def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a registered function with the given arguments.

        Args:
            name: Name of the function to call.
            arguments: Dictionary of arguments to pass to the function.

        Returns:
            Result of the function call.

        Raises:
            KeyError: If the function is not registered.
            ToolCallRefused: If ``authorize`` refused the call.
            ToolArgumentsError: If the arguments don't match the function signature.
        """
        if name not in self.functions:
            raise KeyError(f"Function '{name}' is not registered")

        func_def = self.functions[name]

        if self.authorize is not None:
            await self.authorize(name, arguments)

        # For explicit schema functions, just call directly (validation is external)
        if isinstance(func_def, _ExplicitSchemaFunction):
            return await func_def.function(**arguments)

        # Validate the parameters of introspected functions. An argument the function
        # does not declare is passed on as it came, for a function that takes **kwargs.
        checked = dict(arguments)
        for param in func_def.parameters:
            if param.name not in arguments:
                if param.required:
                    raise ToolArgumentsError(
                        f"Missing required parameter '{param.name}' for function '{name}'"
                    )
                continue
            if param.adapter is None:
                continue
            try:
                checked[param.name] = param.adapter.validate_python(
                    arguments[param.name]
                )
            except ValidationError as invalid:
                problems = "; ".join(error["msg"] for error in invalid.errors())
                raise ToolArgumentsError(
                    f"Invalid value for parameter '{param.name}' of function "
                    f"'{name}': {problems}"
                ) from None

        return await func_def.function(**checked)

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
        if name not in self.functions:
            raise KeyError(f"Function '{name}' is not registered")

        return self.functions[name].function

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


def _adapter_for(type_hint: Any) -> Optional[TypeAdapter]:
    """A validator for an argument's type, or None for a type Pydantic cannot check."""
    try:
        return TypeAdapter(type_hint, config=_ARGUMENT_CONFIG)
    except PydanticSchemaGenerationError:
        return None
    except PydanticUserError:
        # A model or dataclass carries its own config and refuses another.
        try:
            return TypeAdapter(type_hint)
        except PydanticSchemaGenerationError:
            return None


# Global registry instance
function_registry = FunctionRegistry()
