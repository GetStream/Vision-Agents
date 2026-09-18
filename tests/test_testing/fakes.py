"""Scripted LLM and judge doubles for simulation unit tests (no real model)."""

import asyncio
import json
from collections.abc import Callable
from typing import AsyncIterator

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.testing import ChatMessageEvent, JudgeVerdict


def user_says(message: str) -> str:
    return json.dumps({"message": message, "done": False})


def user_done() -> str:
    return json.dumps({"message": "", "done": True})


class ScriptedLLM(LLM):
    """LLM that replays scripted replies, repeating the last one when exhausted.

    Raises ``error`` on the first call instead when it is given.
    """

    def __init__(
        self, replies: list[str], delay: float = 0.0, error: Exception | None = None
    ) -> None:
        super().__init__()
        self.replies = list(replies)
        self.delay = delay
        self.error = error

    @property
    def history(self) -> list[tuple[str, str]]:
        """(role, content) pairs recorded in the conversation given to this LLM."""
        if self._conversation is None:
            return []
        return [(m.role, m.content) for m in self._conversation.messages]

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        if self.error is not None:
            raise self.error
        if self.delay:
            await asyncio.sleep(self.delay)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        yield LLMResponseFinal(text=reply)


class BookingLLM(ScriptedLLM):
    """Agent double that calls its ``book_slot`` tool before every reply."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)

        @self.register_function(description="Book a slot")
        async def book_slot(day: str, time: str) -> dict[str, str]:
            return {"day": day, "time": time, "status": "booked"}

    async def simple_response(
        self,
        text: str,
        participant: Participant | None = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        await self.call_function("book_slot", {"day": "Friday", "time": "10am"})
        async for item in super().simple_response(text, participant):
            yield item


JudgeOutcome = bool | Exception | Callable[[ChatMessageEvent, str], bool]


class ScriptedJudge:
    """Judge that replays outcomes: a verdict, an exception to raise, or a
    predicate over (event, intent). Passes once the script is exhausted."""

    def __init__(self, outcomes: list[JudgeOutcome] | None = None) -> None:
        self.outcomes = list(outcomes or [])

    async def evaluate(self, event: ChatMessageEvent, intent: str) -> JudgeVerdict:
        outcome: JudgeOutcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome(event, intent)
        return JudgeVerdict(success=outcome, reason="scripted")
