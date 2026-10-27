"""The call a function is running for, as the backend that asked for it described it."""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol


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

    Read it inside a function with `get_tool_call_context()`.
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


_current: contextvars.ContextVar[Optional[ToolCall]] = contextvars.ContextVar(
    "tool_call", default=None
)


def get_tool_call_context() -> ToolCall:
    """The call the running function is answering.

    Outside a call, as in a unit test of the function itself or under a model that does
    not say, it is an empty one whose methods do nothing.
    """
    return _current.get() or ToolCall()


@contextlib.contextmanager
def using_tool_call_context(tool_call_context: ToolCall) -> Iterator[ToolCall]:
    """Make `tool_call_context` what `get_tool_call_context()` returns until the block ends.

    The socket that runs a function sets it, and a test of a function can too.
    """
    token = _current.set(tool_call_context)
    try:
        yield tool_call_context
    finally:
        _current.reset(token)
