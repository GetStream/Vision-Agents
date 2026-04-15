import asyncio
import fractions
import os
import uuid
from types import SimpleNamespace
from typing import AsyncIterator, Literal, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from av import VideoFrame
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.huggingface import LLM, VLM
from vision_agents.plugins.huggingface.events import LLMErrorEvent

from conftest import skip_blockbuster


@pytest.fixture()
def huggingface_client_mock():
    mock = AsyncMock()
    mock.chat = MagicMock()
    mock.chat.completions = MagicMock()
    mock.chat.completions.create = AsyncMock()
    return mock


@pytest.fixture()
async def conversation():
    return InMemoryConversation("", [])


@pytest.fixture()
async def llm(huggingface_client_mock, conversation):
    llm_ = LLM(client=huggingface_client_mock, model="test")
    llm_.set_conversation(conversation)
    yield llm_
    await llm_.close()


@pytest.fixture()
async def vlm(huggingface_client_mock, conversation):
    vlm_ = VLM(client=huggingface_client_mock, model="test")
    vlm_.set_conversation(conversation)
    yield vlm_
    await vlm_.close()


def _tc_delta(index: int, tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    """A single tool_call entry inside a streaming delta."""
    return SimpleNamespace(
        index=index,
        id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(
    chunk_id: str,
    content: str = "",
    finish_reason: Optional[
        Literal["stop", "length", "tool_calls", "content_filter"]
    ] = None,
    tool_calls: Optional[list] = None,
) -> SimpleNamespace:
    """A single streaming chat-completion chunk."""
    return SimpleNamespace(
        id=chunk_id,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
    )


class AsyncStreamStub:
    """Async-iterable stub that yields pre-built chunks."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.chunks: list[SimpleNamespace] = []
        self.model = "test"

    def add_chunk(
        self,
        content: str = "",
        finish_reason: Optional[
            Literal["stop", "length", "tool_calls", "content_filter"]
        ] = None,
        tool_calls: Optional[list] = None,
    ):
        self.chunks.append(_chunk(self.id, content, finish_reason, tool_calls))

    async def __aiter__(self) -> AsyncIterator:
        for c in self.chunks:
            yield c


class VideoStreamTrackStub:
    def __init__(self):
        self.frames = []

    async def recv(self):
        try:
            return self._random_video_frame()
        finally:
            await asyncio.sleep(0.0001)

    def _random_video_frame(self, width=800, height=600, format_="bgr24"):
        """Generate a random av.VideoFrame."""
        array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(array, format=format_)
        frame.pts = 0
        frame.time_base = fractions.Fraction(1, 30)
        return frame


class TestHuggingFaceVLM:
    async def test_simple_response_success(
        self, vlm, conversation, huggingface_client_mock
    ):
        track = VideoStreamTrackStub()
        await vlm.watch_video_track(track)

        await conversation.send_message(role="user", user_id="id1", content="message1")
        await conversation.send_message(role="user", user_id="id1", content="message2")

        stream = AsyncStreamStub()
        stream.add_chunk(content="chunk1", finish_reason=None)
        stream.add_chunk(content=" chunk2", finish_reason=None)
        stream.add_chunk(content="", finish_reason="stop")
        huggingface_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        events = []

        @vlm.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent | LLMErrorEvent,
        ):
            events.append(event)

        await asyncio.sleep(2)

        response = await vlm.simple_response(text="prompt")
        await vlm.events.wait(1)
        assert response.text == "chunk1 chunk2"

        assert len(events) == 3
        assert events[0].type == "plugin.llm_response_chunk"
        assert events[0].delta == "chunk1"
        assert events[1].type == "plugin.llm_response_chunk"
        assert events[1].delta == " chunk2"
        assert events[2].type == "plugin.llm_response_completed"

        call_args = huggingface_client_mock.chat.completions.create.call_args_list
        assert len(call_args) == 1
        messages = call_args[0].kwargs["messages"]

        assert len(messages) == 4
        assert messages[0]["content"] == "message1"
        assert messages[1]["content"] == "message2"
        assert messages[2]["content"] == "prompt"
        assert messages[2]["role"] == "user"
        assert messages[3]["content"][0]["type"] == "image_url"

    async def test_simple_response_model_failure(
        self, vlm, conversation, huggingface_client_mock
    ):
        huggingface_client_mock.chat.completions.create = AsyncMock(
            side_effect=OSError("test")
        )

        events = []

        @vlm.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent | LLMErrorEvent,
        ):
            events.append(event)

        await vlm.simple_response(text="prompt")
        await vlm.events.wait(1)
        assert len(events) == 1
        assert events[0].type == "plugin.llm.error"
        assert events[0].error_message == "test"


class TestHuggingFaceLLM:
    async def test_simple_response_success(
        self, llm, conversation, huggingface_client_mock
    ):
        await conversation.send_message(role="user", user_id="id1", content="message1")
        await conversation.send_message(role="user", user_id="id1", content="message2")

        stream = AsyncStreamStub()
        stream.add_chunk(content="chunk1", finish_reason=None)
        stream.add_chunk(content=" chunk2", finish_reason=None)
        stream.add_chunk(content="", finish_reason="stop")
        huggingface_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        events = []

        @llm.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent | LLMErrorEvent,
        ):
            events.append(event)

        response = await llm.simple_response(text="prompt")
        await llm.events.wait(1)
        assert response.text == "chunk1 chunk2"

        assert len(events) == 3
        assert events[0].type == "plugin.llm_response_chunk"
        assert events[0].delta == "chunk1"
        assert events[1].type == "plugin.llm_response_chunk"
        assert events[1].delta == " chunk2"
        assert events[2].type == "plugin.llm_response_completed"

        call_args = huggingface_client_mock.chat.completions.create.call_args_list
        assert len(call_args) == 1
        messages = call_args[0].kwargs["messages"]

        assert len(messages) == 3
        assert messages[0]["content"] == "message1"
        assert messages[1]["content"] == "message2"
        assert messages[2]["content"] == "prompt"
        assert messages[2]["role"] == "user"

    async def test_simple_response_model_failure(
        self, llm, conversation, huggingface_client_mock
    ):
        huggingface_client_mock.chat.completions.create = AsyncMock(
            side_effect=OSError("test")
        )

        events = []

        @llm.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent | LLMErrorEvent,
        ):
            events.append(event)

        await llm.simple_response(text="")
        await llm.events.wait(1)
        assert len(events) == 1
        assert events[0].type == "plugin.llm.error"
        assert events[0].error_message == "test"

    @pytest.mark.integration
    @skip_blockbuster
    async def test_simple_response_huggingface_integration(self, conversation):
        api_key = os.getenv("HF_TOKEN")
        if not api_key:
            pytest.skip("HF_TOKEN not set, skipping integration test")

        llm = LLM(
            api_key=api_key,
            model="meta-llama/Meta-Llama-3-8B-Instruct",
        )
        llm.set_conversation(conversation)

        response = await llm.simple_response(text="Say hello in one word")
        assert response.text


class TestHuggingFaceLLMToolCalling:
    async def test_streaming_tool_calls_parsed_and_executed(
        self, llm, conversation, huggingface_client_mock
    ):
        """Streaming response with finish_reason=tool_calls triggers tool execution."""
        calls_received = []

        @llm.register_function("get_weather", description="Get weather for a city")
        async def get_weather(city: str) -> str:
            calls_received.append(city)
            return "Sunny, 72F"

        # First API call returns a tool_call via streaming
        tool_stream = AsyncStreamStub()
        tool_stream.add_chunk(
            tool_calls=[_tc_delta(0, "call-1", "get_weather", '{"city": "SF"}')]
        )
        tool_stream.add_chunk(finish_reason="tool_calls")

        # Second API call (follow-up after tool execution) returns text
        followup_stream = AsyncStreamStub()
        followup_stream.add_chunk(content="It's sunny in SF!")
        followup_stream.add_chunk(finish_reason="stop")

        huggingface_client_mock.chat.completions.create = AsyncMock(
            side_effect=[tool_stream, followup_stream]
        )

        response = await llm.simple_response(text="What's the weather in SF?")

        assert calls_received == ["SF"]
        assert response.text == "It's sunny in SF!"

        # Verify follow-up call includes tool result messages
        follow_up_call = huggingface_client_mock.chat.completions.create.call_args_list[
            1
        ]
        follow_up_messages = follow_up_call.kwargs["messages"]
        roles = [m["role"] for m in follow_up_messages]
        assert "tool" in roles
        assert "assistant" in roles

    async def test_create_response_passes_tools_to_api(
        self, llm, conversation, huggingface_client_mock
    ):
        """When tools are registered, they are forwarded to the API call."""

        @llm.register_function("lookup", description="Look up info")
        async def lookup(query: str) -> str:
            return "result"

        stream = AsyncStreamStub()
        stream.add_chunk(content="No tool needed.")
        stream.add_chunk(finish_reason="stop")
        huggingface_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        await llm.simple_response(text="hi")

        call_kwargs = huggingface_client_mock.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "lookup" in tool_names


class TestHuggingFaceVLMToolCalling:
    async def test_vlm_streaming_tool_calls(
        self, vlm, conversation, huggingface_client_mock
    ):
        """VLM streaming response with tool calls triggers execution."""
        calls_received = []

        @vlm.register_function("count_objects", description="Count objects in frame")
        async def count_objects(label: str) -> str:
            calls_received.append(label)
            return "3"

        tool_stream = AsyncStreamStub()
        tool_stream.add_chunk(
            tool_calls=[_tc_delta(0, "call-1", "count_objects", '{"label": "person"}')]
        )
        tool_stream.add_chunk(finish_reason="tool_calls")

        followup_stream = AsyncStreamStub()
        followup_stream.add_chunk(content="I see 3 people.")
        followup_stream.add_chunk(finish_reason="stop")

        huggingface_client_mock.chat.completions.create = AsyncMock(
            side_effect=[tool_stream, followup_stream]
        )

        response = await vlm.simple_response(text="How many people?")

        assert calls_received == ["person"]
        assert response.text == "I see 3 people."

    async def test_vlm_create_response_passes_tools(
        self, vlm, conversation, huggingface_client_mock
    ):
        """VLM create_response forwards registered tools to the API."""

        @vlm.register_function("detect", description="Detect objects")
        async def detect(category: str) -> str:
            return "found"

        stream = AsyncStreamStub()
        stream.add_chunk(content="No detection needed.")
        stream.add_chunk(finish_reason="stop")
        huggingface_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        await vlm.create_response(messages=[{"role": "user", "content": "test"}])

        call_kwargs = huggingface_client_mock.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "detect" in tool_names
