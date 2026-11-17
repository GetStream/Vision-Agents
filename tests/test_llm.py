import pytest
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant

from tests.test_agents.test_inference.stubs import LLMStub

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")

# The router agent's line, word for word (`severalSpeakers` in acceleration/internal/agent/agent.go).
SEVERAL_PEOPLE = (
    "Several people are talking in this call. Each of their messages starts with "
    "the name of who said it, then a colon. Never start your reply with such a label."
)


class TestLLM:
    @pytest.mark.parametrize(
        ("instructions", "told"),
        [("be brief", "be brief\n\n" + SEVERAL_PEOPLE), ("", SEVERAL_PEOPLE)],
    )
    async def test_instructions_tell_the_model_once_a_second_person_speaks(
        self, instructions: str, told: str
    ):
        conversation = InMemoryConversation("", [])
        llm = LLMStub()
        llm.set_instructions(instructions)
        assert llm.instructions == instructions, "no conversation yet"

        llm.set_conversation(conversation)
        # Rejoining gives Alice a new participant id; she is still one person.
        rejoined = Participant(original=None, user_id="alice", id="alice-2")
        for participant in (ALICE, rejoined):
            await conversation.upsert_message(
                role="user", user_id="alice", content="Hi", participant=participant
            )
        assert llm.instructions == instructions, "one person"

        await conversation.upsert_message(
            role="user", user_id="bob", content="Me too", participant=BOB
        )
        assert llm.instructions == told
