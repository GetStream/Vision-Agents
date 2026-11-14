import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import AsyncAnthropic, AsyncStream
from anthropic.types import Message as ClaudeMessage
from anthropic.types import TextBlock, ToolUseBlock, Usage
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.plugins.anthropic.anthropic_llm import ClaudeLLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture
async def llm():
    llm = ClaudeLLM(model="claude-sonnet-4-6")
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


class TestClaudeLLM:
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


def _message(content: list, input_tokens: int, output_tokens: int):
    return ClaudeMessage(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=content,
        stop_reason="end_turn",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _text(text: str):
    return TextBlock(type="text", text=text)


def _tool_use():
    return ToolUseBlock(type="tool_use", id="toolu_1", name="probe", input={})


def _events(*events):
    async def iterator():
        for event in events:
            yield event

    stream = AsyncStream(
        cast_to=object,
        response=httpx.Response(200),
        client=AsyncAnthropic(api_key="test"),
    )
    stream._iterator = iterator()
    return stream


def _start(input_tokens: int):
    return SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(usage=SimpleNamespace(input_tokens=input_tokens)),
    )


def _delta(output_tokens: int):
    return SimpleNamespace(
        type="message_delta", usage=SimpleNamespace(output_tokens=output_tokens)
    )


def _tool_events():
    return [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="toolu_1", name="probe"),
        ),
        SimpleNamespace(type="content_block_stop", index=0),
    ]


class TestClaudeLLMUsage:
    async def _run(self, llm: ClaudeLLM, responses: list):
        @llm.register_function(description="probe")
        async def probe() -> str:
            return "ok"

        llm.client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(side_effect=responses)),
            close=AsyncMock(),
        )
        _, final = await collect_simple_response(llm.simple_response("hi"))
        return final

    async def test_non_streaming_usage_is_summed_across_tool_hops(self, llm: ClaudeLLM):
        # initial call (tool_use) -> follow-up hop -> finalization pass
        responses = [
            _message([_tool_use()], 10, 5),
            _message([_text("thinking")], 20, 6),
            _message([_text("done")], 30, 7),
        ]
        final = await self._run_non_streaming(llm, responses)
        assert final.input_tokens == 60
        assert final.output_tokens == 18
        assert final.total_tokens == 78

    async def _run_non_streaming(self, llm: ClaudeLLM, responses: list):
        @llm.register_function(description="probe")
        async def probe() -> str:
            return "ok"

        llm.client = SimpleNamespace(
            messages=SimpleNamespace(create=AsyncMock(side_effect=responses)),
            close=AsyncMock(),
        )
        final = None
        async for item in llm._stream_message(
            messages=[{"role": "user", "content": "hi"}], stream=False
        ):
            final = item
        return final

    async def test_streaming_usage_is_summed_across_tool_hops(self, llm: ClaudeLLM):
        responses = [
            _events(_start(10), *_tool_events(), _delta(5)),
            _events(_start(20), _delta(6)),
            _events(_start(30), _delta(7)),
        ]
        final = await self._run(llm, responses)
        assert final.input_tokens == 60
        assert final.output_tokens == 18
        assert final.total_tokens == 78

    async def test_streaming_usage_without_tools_is_unchanged(self, llm: ClaudeLLM):
        # message_delta reports a running total for the message
        responses = [_events(_start(10), _delta(3), _delta(5))]
        final = await self._run(llm, responses)
        assert final.input_tokens == 10
        assert final.output_tokens == 5


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)
class TestClaudeLLMIntegration:
    """Test suite for ClaudeLLM class with real API calls."""

    async def test_stream(self, llm: ClaudeLLM):
        deltas, final_response = await collect_simple_response(
            llm.simple_response("Explain magma to a 5 year old")
        )
        assert deltas
        assert final_response.text

    async def test_memory(self, llm: ClaudeLLM):
        await collect_simple_response(
            llm.simple_response(text="There are 2 dogs in the room")
        )
        _, response = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )

        assert "8" in response.text or "eight" in response.text

    async def test_tool_calling(self, llm: ClaudeLLM):
        calls: list[str] = []

        @llm.register_function(description="Return a deterministic probe marker.")
        async def probe_tool() -> str:
            calls.append("called")
            return "anthropic_tool_call_probe_ok"

        _, response = await collect_simple_response(
            llm.simple_response(
                "Call the tool named probe_tool now. After receiving the tool result, "
                "reply with only the exact string returned by the tool."
            )
        )

        assert calls == ["called"], "probe_tool was not invoked by Claude"
        assert "anthropic_tool_call_probe_ok" in response.text
