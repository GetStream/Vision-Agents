from dataclasses import dataclass
from typing import AsyncIterator, Literal, Optional, Protocol, runtime_checkable

from ..harness import Harness

RemoteEventType = Literal[
    "participant_joined",
    "participant_left",
    "user_turn_started",
    "user_turn_ended",
    "user_speech",
    "agent_turn_started",
    "agent_speech",
    "agent_turn_ended",
    "error",
    "ended",
]


class RemotePipelineError(Exception):
    """A failure a remote pipeline reported to the agent."""


@dataclass
class RemoteCall:
    """What a remote pipeline needs to know to run a call on the agent's behalf.

    Attributes:
        call_type: The edge provider's call type.
        call_id: The call to join.
        agent_user_id: The identity the agent joins under.
        instructions: The system prompt, already resolved from any `@file.md`.
        harness: How delegated work is configured, if at all.
        cost_tracking: Labels attributed to every request the call makes.
        memory_filter: Which memories this call may recall, keyed by scope.
    """

    call_type: str
    call_id: str
    agent_user_id: str
    instructions: str
    harness: Optional[Harness] = None
    cost_tracking: Optional[dict[str, str]] = None
    memory_filter: Optional[dict[str, str]] = None


@dataclass
class RemoteEvent:
    """Something a remote pipeline did, in the terms the agent thinks in.

    Attributes:
        type: What happened.
        text: What was said, for the speech events.
        user_id: Who said it, empty when it was the agent.
        participant_id: The speaker's participant id, when the pipeline knows it.
        interrupted: Whether an agent turn ended because someone spoke over it.
        error: The failure, for `error` events.
    """

    type: RemoteEventType
    text: str = ""
    user_id: str = ""
    participant_id: str = ""
    interrupted: bool = False
    error: str = ""


@runtime_checkable
class RemotePipeline(Protocol):
    """An LLM that is not a model but a whole pipeline running somewhere else.

    An agent whose LLM satisfies this hands the call over instead of running it: the
    media, the turns, and the speech all happen remotely, and what comes back is a
    stream of events to record. Function calling stays local, since the functions do.
    """

    async def join_remote(self, call: RemoteCall) -> None:
        """Hand the call to the remote pipeline and wait until it is on the call."""
        ...

    def remote_events(self) -> AsyncIterator[RemoteEvent]:
        """Yield events until the call ends."""
        ...

    async def say_remote(self, text: str, interrupt: bool = False) -> None:
        """Speak `text` on the call without asking the model anything."""
        ...

    async def respond_remote(self, text: str, interrupt: bool = True) -> None:
        """Ask the model to reply to an injected instruction."""
        ...

    async def leave_remote(self) -> None:
        """End the remote call. Safe to call after it has already ended."""
        ...
