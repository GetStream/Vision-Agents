import asyncio
import fractions
import os
import uuid
from types import SimpleNamespace
from typing import AsyncIterator, Literal, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from av import VideoFrame
from vision_agents.testing import collect_simple_response
from vision_agents.core.agents.conversation import Conversation, InMemoryConversation
from vision_agents.core.agents.inference.llm_turn import LLMTurn
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM as BaseLLM
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.plugins.huggingface import LLM, VLM

from conftest import skip_blockbuster

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


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


@pytest.fixture
async def huggingface_api() -> AsyncIterator[tuple[str, list[dict]]]:
    """A real server standing in for an OpenAI-compatible endpoint: its base URL, and
    every request body sent.

    Every answer is an empty stream: the tests are about requests.
    """
    bodies: list[dict] = []

    async def stream(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", stream)
    server = TestServer(app)
    await server.start_server()
    yield str(server.make_url("")).rstrip("/"), bodies
    await server.close()


async def answer(
    llm: BaseLLM, conversation: Conversation, participant: Participant, text: str
) -> None:
    """What the agent does with a participant's words: keep them, then ask the LLM."""
    await conversation.upsert_message(
        role="user", user_id=participant.user_id, content=text, participant=participant
    )
    events = EventManager()
    events.register(LLMResponseFinalEvent)
    turn = LLMTurn(
        transcript=text,
        participant=participant,
        events=events,
        speaker=conversation.speaker_label(participant),
    )
    turn.start(llm)
    await turn.stream.collect()
    await conversation.send_message(role="assistant", user_id="agent", content="Ok")


async def assert_who_spoke_is_named(
    llm: BaseLLM, conversation: Conversation, bodies: list[dict]
) -> None:
    """Two turns of one person go as they did, then the turns of two are named."""
    llm.set_conversation(conversation)

    await answer(llm, conversation, ALICE, "Hi")
    await answer(llm, conversation, ALICE, "Are you there?")
    # One person: what the base sha (486427184) sent for the same two turns.
    assert bodies[-1]["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Ok"},
        {"role": "user", "content": "Are you there?"},
    ]

    await answer(llm, conversation, BOB, "Me too")
    await answer(llm, conversation, ALICE, "Who is that?")
    assert bodies[-1]["messages"][2:] == [
        {"role": "user", "content": "Are you there?"},
        {"role": "assistant", "content": "Ok"},
        {"role": "user", "content": "Bob: Me too"},
        {"role": "assistant", "content": "Ok"},
        {"role": "user", "content": "Alice: Who is that?"},
    ]


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
    async def test_who_spoke_is_named_once_a_second_person_has(
        self, huggingface_api: tuple[str, list[dict]]
    ):
        base_url, bodies = huggingface_api
        vlm = VLM(model="test", base_url=base_url, api_key="test")
        await assert_who_spoke_is_named(vlm, InMemoryConversation("", []), bodies)
        await vlm.close()

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

        await asyncio.sleep(2)

        deltas, final = await collect_simple_response(
            vlm.simple_response(text="prompt")
        )

        assert final.text == "chunk1 chunk2"
        assert [d.delta for d in deltas] == ["chunk1", " chunk2"]

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

        deltas, final = await collect_simple_response(
            vlm.simple_response(text="prompt")
        )

        assert deltas == []
        assert final.text == ""


class TestHuggingFaceLLM:
    async def test_who_spoke_is_named_once_a_second_person_has(
        self, huggingface_api: tuple[str, list[dict]]
    ):
        base_url, bodies = huggingface_api
        llm = LLM(model="test", base_url=base_url, api_key="test")
        await assert_who_spoke_is_named(llm, InMemoryConversation("", []), bodies)
        await llm.close()

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

        deltas, final = await collect_simple_response(
            llm.simple_response(text="prompt")
        )

        assert final.text == "chunk1 chunk2"
        assert [d.delta for d in deltas] == ["chunk1", " chunk2"]

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

        deltas, final = await collect_simple_response(llm.simple_response(text=""))

        assert deltas == []
        assert final.text == ""

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

        _, final = await collect_simple_response(
            llm.simple_response(text="Say hello in one word")
        )
        assert final.text


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

        _, final = await collect_simple_response(
            llm.simple_response(text="What's the weather in SF?")
        )

        assert calls_received == ["SF"]
        assert final.text == "It's sunny in SF!"

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

        await collect_simple_response(llm.simple_response(text="hi"))

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

        _, final = await collect_simple_response(
            vlm.simple_response(text="How many people?")
        )

        assert calls_received == ["person"]
        assert final.text == "I see 3 people."

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

        await collect_simple_response(
            vlm.create_response(messages=[{"role": "user", "content": "test"}])
        )

        call_kwargs = huggingface_client_mock.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        assert "detect" in tool_names
