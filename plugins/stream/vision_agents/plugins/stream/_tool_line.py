"""What the sockets that deliver tool calls share: the call a function sees, and its line back."""

import asyncio
from typing import Any, Awaitable, Callable, Optional

from vision_agents.core.llm.function_registry import FunctionRegistry
from vision_agents.core.llm.tool_call import (
    ToolCall,
    ToolConversation,
    ToolUser,
    calling,
)

Send = Callable[[dict[str, Any]], Awaitable[None]]


class Holds:
    """The calls that asked to be held, waiting on the router's tool_held or tool_cancel."""

    def __init__(self) -> None:
        self._waiting: dict[str, asyncio.Future[bool]] = {}
        self._held: set[str] = set()

    async def ask(self, call_id: str, send: Callable[[], Awaitable[None]]) -> bool:
        """Send the hold and wait for the router to say whether the call was."""
        answer: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._waiting[call_id] = answer
        try:
            await send()
            held = await answer
        finally:
            self._waiting.pop(call_id, None)
        if held:
            self._held.add(call_id)
        return held

    def answer(self, call_id: str, held: bool) -> None:
        """Hand the router's answer to the call waiting on it, if one is."""
        waiting = self._waiting.get(call_id)
        if waiting is not None and not waiting.done():
            waiting.set_result(held)

    def held(self, call_id: str) -> bool:
        """Whether the router keeps this call going when the caller talks over the agent."""
        return call_id in self._held

    def forget(self, call_id: str) -> None:
        self._held.discard(call_id)


class SocketLine:
    """A call's way back to its session, over the socket that delivered it."""

    def __init__(self, send: Send, holds: Holds) -> None:
        self._send = send
        self._holds = holds

    async def update(self, call_id: str, text: str, verbatim: bool) -> None:
        await self._send(
            {
                "type": "tool_update",
                "tool_call_id": call_id,
                "text": text,
                "verbatim": verbatim,
            }
        )

    async def hold(self, call_id: str) -> bool:
        return await self._holds.ask(
            call_id, lambda: self._send({"type": "tool_hold", "tool_call_id": call_id})
        )


def cancel(call_id: str, holds: Holds, task: Optional[asyncio.Task]) -> None:
    """Act on a tool_cancel: stop the call unless it asked to be held and was."""
    holds.answer(call_id, False)
    if task is not None and not holds.held(call_id):
        task.cancel()


def call_of(frame: dict[str, Any], session_id: str, line: SocketLine) -> ToolCall:
    """The call a tool_call frame asks for, as `current_call()` gives it to the function.

    `session_id` stands in for a frame that does not name its session, as one from a
    router from before it said does not.
    """
    user = frame.get("user") or {}
    conversation = frame.get("conversation") or {}
    return ToolCall(
        id=str(frame.get("id", "")),
        session_id=str(frame.get("session_id") or session_id),
        turn_id=str(frame.get("turn_id", "")),
        user=ToolUser(
            id=str(user.get("id", "")),
            name=str(user.get("name", "")),
            custom=dict(user.get("custom") or {}),
        ),
        conversation=ToolConversation(
            cid=str(conversation.get("cid", "")),
            custom=dict(conversation.get("custom") or {}),
        ),
        approved=frame.get("approved") is True,
        line=line,
    )


async def run_call(
    functions: FunctionRegistry,
    name: str,
    arguments: dict[str, Any],
    call: ToolCall,
    holds: Holds,
) -> Any:
    """Run one function for `call`, within its own timeout when it has one."""
    timeout = functions.policy(name).timeout
    try:
        with calling(call):
            if not timeout:
                return await functions.call_function(name, arguments)
            try:
                return await asyncio.wait_for(
                    functions.call_function(name, arguments), timeout
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"{name} took longer than {timeout:g}s") from None
    finally:
        holds.forget(call.id)
