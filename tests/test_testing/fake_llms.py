"""Fake LLMs for exercising ``vision_agents.testing`` without a real model."""

from typing import AsyncIterator, Optional

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem


class ScriptedLLM(LLM):
    """Returns a fixed reply and records every prompt it receives.

    Raises ``error`` instead of replying when it is given.
    """

    def __init__(self, reply: str, error: Exception | None = None) -> None:
        super().__init__()
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        self.prompts.append(text)
        if self.error is not None:
            raise self.error
        yield LLMResponseFinal(text=self.reply)


class ToolCallingLLM(LLM):
    """Runs scripted tool calls through the base-class tool loop, then replies.

    ``script`` maps user text to the tool calls to execute for that turn.
    """

    def __init__(
        self,
        script: dict[str, list[NormalizedToolCallItem]] | None = None,
        reply: str = "done",
    ) -> None:
        super().__init__()
        self.script = script or {}
        self.reply = reply

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        calls = self.script.get(text, [])
        if calls:
            await self._dedup_and_execute(calls)
        yield LLMResponseFinal(text=self.reply)
