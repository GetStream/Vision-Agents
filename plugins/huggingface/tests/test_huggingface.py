import asyncio
import fractions
import os
import uuid
from typing import AsyncIterator, Literal, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from av import VideoFrame
from conftest import skip_blockbuster
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.huggingface import LLM, VLM
from vision_agents.plugins.huggingface.events import LLMErrorEvent


@pytest.fixture()
def huggingface_client_mock():
    mock = MagicMock()
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
    return llm_


@pytest.fixture()
async def vlm(huggingface_client_mock, conversation):
    vlm_ = VLM(client=huggingface_client_mock, model="test")
    vlm_.set_conversation(conversation)
    return vlm_


class ChatCompletionChunkMock:
    """Mock for HuggingFace chat completion chunks."""

    def __init__(
        self,
        chunk_id: str,
        content: str = "",
        finish_reason: Optional[
            Literal["stop", "length", "tool_calls", "content_filter"]
        ] = None,
    ):
        self.id = chunk_id
        self.choices = [
            MagicMock(
                delta=MagicMock(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ]


class AsyncStreamStub:
    """Mock of async streaming response."""

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.chunks = []
        self.model = "test"

    def add_chunk(
        self,
        content: str = "",
        finish_reason: Optional[
            Literal["stop", "length", "tool_calls", "content_filter"]
        ] = None,
    ):
        self.chunks.append(
            ChatCompletionChunkMock(
                chunk_id=self.id,
                content=content,
                finish_reason=finish_reason,
            )
        )

    async def __aiter__(self) -> AsyncIterator:
        for chunk in self.chunks:
            yield chunk


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
            side_effect=ValueError("test")
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
            side_effect=ValueError("test")
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
