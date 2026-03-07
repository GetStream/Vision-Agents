"""Unit tests for LLMJudge."""

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.testing import ChatMessageEvent, LLMJudge
from vision_agents.testing._judge import _JUDGE_SYSTEM_PROMPT


class _FakeLLM(LLM):
    """Minimal LLM that returns a canned JSON verdict."""

    def __init__(self, response_text: str = '{"verdict": "pass", "reason": "ok"}'):
        super().__init__()
        self._response_text = response_text

    async def simple_response(
        self,
        text: str = "",
        processors: list[Processor] | None = None,
        participant: Participant | None = None,
    ) -> LLMResponseEvent:
        return LLMResponseEvent(original=None, text=self._response_text)


class TestLLMJudge:
    async def test_sets_instructions_at_init(self):
        llm = _FakeLLM()
        LLMJudge(llm)
        assert llm._instructions == _JUDGE_SYSTEM_PROMPT

    async def test_does_not_mutate_instructions_during_evaluate(self):
        llm = _FakeLLM()
        judge = LLMJudge(llm)

        assert llm._instructions == _JUDGE_SYSTEM_PROMPT

        event = ChatMessageEvent(role="assistant", content="Hello!")
        await judge.evaluate(event, intent="Greeting")

        assert llm._instructions == _JUDGE_SYSTEM_PROMPT
