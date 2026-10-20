import os
from typing import AsyncIterator

import grpc
import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS
from vision_agents.plugins.xai.llm import XAILLM
from xai_sdk import AsyncClient
from xai_sdk.proto import chat_pb2, chat_pb2_grpc

from vision_agents.testing import answer, collect_simple_response

load_dotenv()

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


@pytest.fixture
async def xai_api() -> AsyncIterator[
    tuple[AsyncClient, list[chat_pb2.GetCompletionsRequest]]
]:
    """A real gRPC server standing in for xAI: a client of it, and every request sent.

    Every answer is one streamed "Ok": the tests are about requests.
    """
    requests: list[chat_pb2.GetCompletionsRequest] = []

    class Chat(chat_pb2_grpc.ChatServicer):
        async def GetCompletionChunk(self, request, context):
            requests.append(request)
            yield chat_pb2.GetChatCompletionChunk(
                id="r1",
                outputs=[
                    chat_pb2.CompletionOutputChunk(
                        delta=chat_pb2.Delta(
                            content="Ok", role=chat_pb2.MessageRole.ROLE_ASSISTANT
                        ),
                        finish_reason="REASON_STOP",
                    )
                ],
            )

    server = grpc.aio.server()
    chat_pb2_grpc.add_ChatServicer_to_server(Chat(), server)
    port = server.add_insecure_port("localhost:0")
    await server.start()
    client = AsyncClient(
        api_key="test", api_host=f"localhost:{port}", use_insecure_channel=True
    )
    yield client, requests
    await server.stop(None)


class TestXAILLM:
    async def test_a_chat_opened_after_two_people_spoke_tells_the_model(
        self, xai_api: tuple[AsyncClient, list[chat_pb2.GetCompletionsRequest]]
    ):
        client, requests = xai_api
        conversation = InMemoryConversation("", [])
        llm = XAILLM(model="test", client=client)
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")
        # The chat holds the instructions it was opened with; this one opens once Alice
        # and Bob have both spoken.
        await conversation.upsert_message(
            role="user", user_id="alice", content="Hi", participant=ALICE
        )

        await answer(llm, conversation, BOB, "Me too")

        system = requests[-1].messages[0]
        assert system.role == chat_pb2.MessageRole.ROLE_SYSTEM
        assert system.content[0].text == "be brief\n\n" + SEVERAL_SPEAKERS
        assert requests[-1].messages[-1].content[0].text == "Bob: Me too"

    async def test_a_chat_opened_by_one_person_is_told_the_instructions(
        self, xai_api: tuple[AsyncClient, list[chat_pb2.GetCompletionsRequest]]
    ):
        client, requests = xai_api
        conversation = InMemoryConversation("", [])
        llm = XAILLM(model="test", client=client)
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")

        await answer(llm, conversation, ALICE, "Hi")

        # What the base sha (b5a81650b) sent.
        assert [m.content[0].text for m in requests[-1].messages] == ["be brief", "Hi"]


@pytest.fixture
async def llm():
    llm = XAILLM(model="grok-4-latest", api_key=os.getenv("XAI_API_KEY"))
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("XAI_API_KEY"), reason="XAI_API_KEY not set")
class TestXAILLMIntegration:
    """Test suite for XAILLM class with live API calls."""

    async def test_simple_response(self, llm: XAILLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Explain quantum computing in 1 paragraph")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: XAILLM):
        await collect_simple_response(
            llm.simple_response(text="There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response(text="How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_tool_calling(self, llm: XAILLM):
        @llm.register_function()
        async def get_weather(location: str) -> str:
            """Get the weather for a location."""
            return f"The weather in {location} is sunny."

        _, final = await collect_simple_response(
            llm.simple_response("What is the weather in San Francisco?")
        )

        assert "sunny" in final.text.lower()
