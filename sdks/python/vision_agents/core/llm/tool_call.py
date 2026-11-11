"""The call a function is running for, as the backend that asked for it described it."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, get_type_hints


class CallNotHeld(RuntimeError):
    """The call was stopped before `disallow_interruptions` could keep it going."""


class ToolLine(Protocol):
    """How a running call reaches the session it runs for.

    The socket that delivered the call provides one. Without one, as outside a call,
    `ToolCall.update`, `say` and `disallow_interruptions` do nothing.
    """

    async def update(self, call_id: str, text: str, verbatim: bool) -> None:
        """Say how the call is going, or with `verbatim`, words to say as written."""

    async def hold(self, call_id: str) -> bool:
        """Ask for the call to keep going when the caller talks over the agent.

        Returns whether it was still running to be held.
        """


@dataclass(frozen=True)
class ToolUser:
    """The person whose turn asked for a call."""

    id: str = ""
    name: str = ""
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolConversation:
    """The Chat channel a call was made in, and the session's own labels."""

    cid: str = ""
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """The call a function is running for.

    A function whose first parameter is annotated `ToolCall` is handed it there:

        @llm.register_function(description="Book a flight")
        async def book_flight(tool_call_context: ToolCall, flight_id: str) -> str:
            await tool_call_context.update("checking seats")
            return await bookings.book(flight_id, key=tool_call_context.id)

    Outside a call, as under a model that does not say, it is an empty one whose methods
    do nothing.
    """

    # id is the id the model gave the call. A call sent again keeps it, so it is the key to
    # make a side effect happen once.
    id: str = ""
    session_id: str = ""
    turn_id: str = ""
    # user is the person whose turn asked for it, empty when the backend did not say.
    user: ToolUser = field(default_factory=ToolUser)
    conversation: ToolConversation = field(default_factory=ToolConversation)
    # approved says the caller already said yes to this call out loud, when the function
    # was registered with `needs_approval` and the agent asked them on a voice call.
    approved: bool = False
    line: Optional[ToolLine] = field(default=None, repr=False, compare=False)

    async def update(self, text: str) -> None:
        """Say how the call is going.

        A voice agent tells the caller in its own words once nobody is talking, and a
        conversation shows it on the call's step until the call ends. A newer update
        replaces one not yet said, and the result replaces both.
        """
        if self.line is not None and self.id:
            await self.line.update(self.id, text, False)

    async def say(self, text: str) -> None:
        """Have the agent say `text` as written while the call runs.

        A text conversation posts it as a message. A speech-to-speech model is asked for
        these words and may paraphrase them.
        """
        if self.line is not None and self.id:
            await self.line.update(self.id, text, True)

    async def disallow_interruptions(self) -> None:
        """Keep the call going from now on when the caller talks over the agent.

        Call it before a write that must not be left halfway. From then on only the
        function's own timeout and its connection ending stop it.

        Raises:
            CallNotHeld: The call had already been stopped, so the write should not start.
        """
        if self.line is None or not self.id:
            return
        if not await self.line.hold(self.id):
            raise CallNotHeld(f"the call {self.id} was stopped before it could be held")


def context_parameter(func: Callable[..., Any]) -> Optional[str]:
    """The name of `func`'s first parameter when it is annotated `ToolCall`, else None.

    That parameter is handed the call rather than filled in by the model.
    """
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return None
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        return None
    first = parameters[0]
    annotation = first.annotation
    if isinstance(annotation, str):
        try:
            annotation = get_type_hints(func).get(first.name)
        except Exception:
            return None
    if inspect.isclass(annotation) and issubclass(annotation, ToolCall):
        return first.name
    return None
