import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.anthropic.anthropic_llm import ClaudeLLM

load_dotenv()


@pytest.fixture
async def llm() -> ClaudeLLM:
    llm = ClaudeLLM(model="claude-sonnet-4-6")
    llm.set_conversation(InMemoryConversation("be friendly", []))
    return llm


class TestClaudeLLM:
    """Test suite for ClaudeLLM class with real API calls."""

    async def test_message(self, llm: ClaudeLLM):
        messages = ClaudeLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    async def test_advanced_message(self, llm: ClaudeLLM):
        advanced = {
            "role": "user",
            "content": "Explain quantum entanglement in simple terms.",
        }
        messages2 = ClaudeLLM._normalize_message(advanced)
        assert messages2[0].original is not None

    @pytest.mark.integration
    async def test_simple(self, llm: ClaudeLLM):
        response = await llm.simple_response(
            "Explain quantum computing in 1 paragraph",
        )
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: ClaudeLLM):
        response = await llm.create_message(
            messages=[{"role": "user", "content": "say hi"}],
            max_tokens=1000,
        )

        # Assertions
        assert response.text

    @pytest.mark.integration
    async def test_stream(self, llm: ClaudeLLM):
        streamingWorks = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streamingWorks
            streamingWorks = True

        await llm.simple_response("Explain magma to a 5 year old")
        # Wait for all events in queue to be processed
        await llm.events.wait()

        assert streamingWorks

    @pytest.mark.integration
    async def test_memory(self, llm: ClaudeLLM):
        await llm.simple_response(
            text="There are 2 dogs in the room",
        )
        response = await llm.simple_response(
            text="How many paws are there in the room?",
        )

        assert "8" in response.text or "eight" in response.text

    @pytest.mark.integration
    async def test_native_memory(self, llm: ClaudeLLM):
        await llm.create_message(
            messages=[{"role": "user", "content": "There are 2 dogs in the room"}],
            max_tokens=1000,
        )
        response = await llm.create_message(
            messages=[
                {"role": "user", "content": "How many paws are there in the room?"}
            ],
            max_tokens=1000,
        )
        assert "8" in response.text or "eight" in response.text

    def test_merge_messages_alternating_roles_unchanged(self, llm: ClaudeLLM):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        assert llm._merge_messages(messages) == messages

    def test_merge_messages_identical_consecutive_collapses(self, llm: ClaudeLLM):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},
        ]
        result = llm._merge_messages(messages)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_merge_messages_different_content_produces_blocks(self, llm: ClaudeLLM):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = llm._merge_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]

    def test_merge_messages_list_content_merges(self, llm: ClaudeLLM):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": "b"},
        ]
        result = llm._merge_messages(messages)
        assert len(result) == 1
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_merge_messages_empty_input(self, llm: ClaudeLLM):
        assert llm._merge_messages([]) == []

    def test_normalize_message_string_content(self, llm: ClaudeLLM):
        messages = ClaudeLLM._normalize_message({"role": "user", "content": "hello"})
        assert len(messages) == 1
        assert messages[0].content == "hello"
        assert messages[0].role == "user"

    def test_normalize_message_list_content_stringified(self, llm: ClaudeLLM):
        messages = ClaudeLLM._normalize_message(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "text", "text": "world"},
                ],
            }
        )
        assert len(messages) == 1
        assert messages[0].content == "hello world"
        assert isinstance(messages[0].content, str)
        assert messages[0].role == "assistant"
