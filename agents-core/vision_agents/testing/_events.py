"""Event types captured during a test run.

These normalized events represent what happened during a single
conversation turn: messages, function calls, and their outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ChatMessageEvent:
    """A chat message produced during a turn."""

    role: str
    content: str
    type: Literal["message"] = field(default="message", init=False)


@dataclass
class FunctionCallEvent:
    """The agent requested a tool/function call."""

    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
    type: Literal["function_call"] = field(default="function_call", init=False)


@dataclass
class FunctionCallOutputEvent:
    """The result of a tool/function call."""

    name: str
    output: Any
    is_error: bool = False
    tool_call_id: str | None = None
    execution_time_ms: float | None = None
    type: Literal["function_call_output"] = field(
        default="function_call_output", init=False
    )


RunEvent = ChatMessageEvent | FunctionCallEvent | FunctionCallOutputEvent
