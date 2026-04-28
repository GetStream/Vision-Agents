import asyncio
import fractions
import os
import time
import uuid
from typing import AsyncIterator, Literal, Optional
from unittest.mock import AsyncMock

import numpy as np
import openai
import pytest
from av import VideoFrame
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
)
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.openai import ChatCompletionsLLM, ChatCompletionsVLM

from tests.utils import collect_simple_response


@pytest.fixture()
def openai_client_mock():
    return AsyncMock(spec_set=openai.AsyncClient)


@pytest.fixture()
async def conversation():
    return InMemoryConversation("", [])


@pytest.fixture()
async def llm(openai_client_mock, conversation):
    llm_ = ChatCompletionsLLM(client=openai_client_mock, model="test")
    llm_.set_conversation(conversation)
    return llm_


@pytest.fixture()
async def vlm(openai_client_mock, conversation):
    llm_ = ChatCompletionsVLM(client=openai_client_mock, model="test")
    llm_.set_conversation(conversation)
    return llm_


class TestChatCompletionsVLM:
    async def test_simple_response_success(self, vlm, conversation, openai_client_mock):
        # Mock video track
        track = VideoStreamTrackStub()
        await vlm.watch_video_track(track)

        # Mock conversation
        await conversation.send_message(role="user", user_id="id1", content="message1")
        await conversation.send_message(role="user", user_id="id1", content="message2")

        # Mock model response
        stream = AsyncStreamStub()
        stream.add_chunk(content="chunk1", finish_reason=None)
        stream.add_chunk(content=" chunk2", finish_reason=None)
        stream.add_chunk(content="", finish_reason="stop")
        openai_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        # Wait a few seconds to let the video forwarder consume video frames
        await asyncio.sleep(2)

        deltas, final = await collect_simple_response(vlm.simple_response(text="prompt"))
        assert final.text == "chunk1 chunk2"
        assert [d.delta for d in deltas] == ["chunk1", " chunk2"]

        await vlm.stop_watching_video_track()

        # Check that correct messages are sent to the model
        call_args = openai_client_mock.chat.completions.create.call_args_list
        assert len(call_args) == 1
        messages = call_args[0].kwargs["messages"]

        # Must send three conversation messages and one message with frames
        assert len(messages) == 4
        assert messages[0]["content"] == "message1"
        assert messages[1]["content"] == "message2"
        assert messages[2]["content"] == "prompt"
        assert messages[2]["role"] == "user"
        assert messages[3]["content"][0]["type"] == "image_url"

    async def test_simple_response_model_failure(
        self, vlm, conversation, openai_client_mock
    ):
        openai_client_mock.chat.completions.create = AsyncMock(
            side_effect=ValueError("test")
        )

        _, final = await collect_simple_response(vlm.simple_response(text="prompt"))
        assert final.text == ""
        assert final.original is None


class TestChatCompletionsLLM:
    async def test_simple_response_success(self, llm, conversation, openai_client_mock):
        # Mock conversation
        await conversation.send_message(role="user", user_id="id1", content="message1")
        await conversation.send_message(role="user", user_id="id1", content="message2")

        # Mock model response
        stream = AsyncStreamStub()
        stream.add_chunk(content="chunk1", finish_reason=None)
        stream.add_chunk(content=" chunk2", finish_reason=None)
        stream.add_chunk(content="", finish_reason="stop")
        openai_client_mock.chat.completions.create = AsyncMock(return_value=stream)

        deltas, final = await collect_simple_response(
            llm.simple_response(text="prompt")
        )
        assert final.text == "chunk1 chunk2"
        assert [delta.delta for delta in deltas] == ["chunk1", " chunk2"]

        # Check that correct messages are sent to the model
        call_args = openai_client_mock.chat.completions.create.call_args_list
        assert len(call_args) == 1
        messages = call_args[0].kwargs["messages"]

        # Must send three conversation messages
        assert len(messages) == 3
        assert messages[0]["content"] == "message1"
        assert messages[1]["content"] == "message2"
        assert messages[2]["content"] == "prompt"
        assert messages[2]["role"] == "user"

    async def test_simple_response_model_failure(
        self, llm, conversation, openai_client_mock
    ):
        openai_client_mock.chat.completions.create = AsyncMock(
            side_effect=ValueError("test")
        )

        _, final = await collect_simple_response(llm.simple_response(text=""))
        assert final.text == ""
        assert final.original is None

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.getenv("BASETEN_BASE_URL"),
        reason="BASETEN_BASE_URL not set, skipping test",
    )
    @pytest.mark.skipif(
        not os.getenv("BASETEN_API_KEY"),
        reason="BASETEN_API_KEY not set, skipping test",
    )
    async def test_simple_response_model_baseten_deepseek(self, conversation):
        api_key = os.getenv("BASETEN_API_KEY")
        base_url = os.getenv("BASETEN_BASE_URL")

        llm = ChatCompletionsLLM(
            api_key=api_key, base_url=base_url, model="deepseek-ai/DeepSeek-V3.1"
        )
        llm.set_conversation(conversation)

        deltas, final = await collect_simple_response(
            llm.simple_response(text="greet the user")
        )
        assert deltas
        assert final.text


class AsyncStreamStub:
    """
    Mock of AsyncStream[ChatCompletionChunk]
    """

    def __init__(self):
        self.id = str(uuid.uuid4())
        self.chunks = []
        self.model = "test"

    def add_chunk(
        self,
        content: str = "",
        finish_reason: Optional[
            Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        ] = None,
    ):
        choice = Choice(
            index=0,
            finish_reason=finish_reason,
            delta=ChoiceDelta(content=content, role="assistant"),
        )
        self.chunks.append(
            ChatCompletionChunk(
                object="chat.completion.chunk",
                id=self.id,
                model=self.model,
                created=int(time.time()),
                choices=[choice],
            )
        )

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
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
        """
        Generate a random av.VideoFrame.
        """
        # Random pixel data
        array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Create a VideoFrame from the array
        frame = VideoFrame.from_ndarray(array, format=format_)

        # Optionally set timing metadata
        frame.pts = 0
        frame.time_base = fractions.Fraction(1, 30)  # simulate 30fps
        return frame
