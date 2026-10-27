"""
Function registry for managing available functions that can be called by LLMs.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Type, get_type_hints
from dataclasses import dataclass, field

from .llm_types import ToolSchema
from .tool_arguments import Arguments


@dataclass
class FunctionParameter:
    """Represents a parameter of a function."""

    name: str
    type: Type
    description: Optional[str] = None
    required: bool = True
    default: Any = None


# A {name} in an approval question, filled in from the call's arguments.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class ToolPolicy:
    """How a backend that runs the agent elsewhere treats calls to one function.

    A model run in this process ignores it.
    """

    # interruptible lets the caller talking over the agent cancel a call in flight. Off, a
    # call keeps running and the model is told it is still going.
    interruptible: bool = False
    # timeout is how long one call may take, in seconds. Zero takes the session's.
    timeout: float = 0.0
    # approval is the question a person answers before each call runs, with {name} for an
    # argument. Empty runs calls as they come.
    approval: str = ""


@dataclass
class FunctionDefinition:
    """Represents a complete function definition."""

    name: str
    description: str
    parameters: List[FunctionParameter]
    function: Callable
    returns: Optional[Type] = None
    policy: ToolPolicy = ToolPolicy()
    # What the model is told the function takes, and what each call's arguments are parsed
    # by before it runs.
    arguments: Arguments = field(kw_only=True)


@dataclass
class _ExplicitSchemaFunction:
    """Represents a function with an explicitly provided JSON schema (e.g., MCP tools)."""

    name: str
    description: str
    parameters_schema: Dict[str, Any]
    function: Callable
    policy: ToolPolicy = ToolPolicy()


class FunctionRegistry:
    """Registry for managing available functions that can be called by LLMs."""

    def __init__(self):
        self.functions: Dict[str, FunctionDefinition | _ExplicitSchemaFunction] = {}

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict[str, Any]] = None,
        interruptible: bool = False,
        timeout: float = 0.0,
        needs_approval: str = "",
    ) -> Callable:
        """
        Decorator to register a function with the registry.

        Args:
            name: Optional custom name for the function. If not provided, uses the function name.
            description: Optional description for the function. If not provided, uses the docstring.
            parameters_schema: Optional explicit JSON schema for parameters. If provided, this is
                used instead of inferring from type hints. Useful for MCP tools or external APIs.
            interruptible: Let the caller talking over the agent cancel a call in flight. Off,
                the call keeps running. Applies where Stream runs the agent.
            timeout: How long one call may take, in seconds, in place of the session's.
                Zero takes the session's. Applies where Stream runs the agent.
            needs_approval: A question a person answers before each call runs, such as
                "Book flight {flight_id}?". Each {name} is an argument, filled in from the
                call. On a voice call the agent asks it out loud and waits 15 seconds for a
                yes; in writing the question goes to the app. Applies where Stream runs the
                agent.

        Returns:
            Decorator function.

        Raises:
            ValueError: The function is synchronous, the timeout is negative, or the
                approval question is blank or names an argument the function does not take.
        """
        if timeout < 0:
            raise ValueError(f"a timeout cannot be negative, got {timeout}")
        approval = needs_approval.strip()
        if needs_approval and not approval:
            raise ValueError("an approval question cannot be blank")
        policy = ToolPolicy(
            interruptible=interruptible, timeout=timeout, approval=approval
        )

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
                _check_approval(
                    func_name, approval, parameters_schema.get("properties") or {}
                )
                self.functions[func_name] = _ExplicitSchemaFunction(
                    name=func_name,
                    description=func_description,
                    parameters_schema=parameters_schema,
                    function=func,
                    policy=policy,
                )
                return func

            # Otherwise, extract from type hints
            type_hints = get_type_hints(func, include_extras=True)
            sig = inspect.signature(func)
            arguments = Arguments(func_name, func, type_hints)

            parameters = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_type = type_hints.get(param_name, type(None))
                param_description = None

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

            _check_approval(func_name, approval, arguments.names)
            function_def = FunctionDefinition(
                name=func_name,
                description=func_description,
                parameters=parameters,
                function=func,
                returns=return_type,
                policy=policy,
                arguments=arguments,
            )

            self.functions[func_name] = function_def
            return func

        return decorator

    def get_function(
        self, name: str
    ) -> Optional[FunctionDefinition | _ExplicitSchemaFunction]:
        """Get a function definition by name."""
        return self.functions.get(name)

    def policy(self, name: str) -> ToolPolicy:
        """How calls to the function `name` are treated, or the default for one not here."""
        function = self.functions.get(name)
        return function.policy if function is not None else ToolPolicy()

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
            TypeError: If the arguments don't match the function signature.
        """
        if name not in self.functions:
            raise KeyError(f"Function '{name}' is not registered")

        func_def = self.functions[name]

        # For explicit schema functions, just call directly (validation is external)
        if isinstance(func_def, _ExplicitSchemaFunction):
            return await func_def.function(**arguments)

        return await func_def.function(**func_def.arguments.parse(arguments))

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
        return ToolSchema(
            name=func_def.name,
            description=func_def.description,
            parameters_schema=func_def.arguments.json_schema(),
        )


def _check_approval(name: str, approval: str, arguments: Any) -> None:
    """Refuse an approval question that names an argument `name` does not take."""
    for placeholder in _PLACEHOLDER.findall(approval):
        if placeholder not in arguments:
            raise ValueError(
                f"the approval question for {name} names {{{placeholder}}}, "
                f"which is not one of its arguments"
            )


# Global registry instance
function_registry = FunctionRegistry()
