"""Tests for AWS plugin."""

import os
from typing import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS
from vision_agents.plugins.aws.aws_llm import BedrockLLM

from vision_agents.testing import answer, collect_simple_response

load_dotenv()

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


@pytest.fixture
async def bedrock_api(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[list[dict]]:
    """A real server standing in for Bedrock, and every request body sent to it.

    Every answer is an empty event stream: the tests are about requests.
    """
    bodies: list[dict] = []

    async def converse_stream(request: web.Request) -> web.Response:
        bodies.append(await request.json())
        return web.Response(content_type="application/vnd.amazon.eventstream")

    app = web.Application()
    app.router.add_post("/model/{model}/converse-stream", converse_stream)
    server = TestServer(app)
    await server.start_server()
    # boto3 sends a service's requests to AWS_ENDPOINT_URL_<SERVICE ID> when it is set:
    # https://docs.aws.amazon.com/sdkref/latest/guide/feature-ss-endpoints.html
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", str(server.make_url("")).rstrip("/")
    )
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    yield bodies
    await server.close()


class TestBedrockLLM:
    async def test_who_spoke_is_named_once_a_second_person_has(
        self, bedrock_api: list[dict]
    ):
        conversation = InMemoryConversation("", [])
        llm = BedrockLLM(
            model="test", aws_access_key_id="test", aws_secret_access_key="test"
        )
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")

        await answer(llm, conversation, ALICE, "Hi")
        await answer(llm, conversation, ALICE, "Are you there?")
        # One person: what the base sha (9653cb696) sent for the same two turns.
        assert bedrock_api[-1]["system"] == [{"text": "be brief"}]
        assert bedrock_api[-1]["messages"] == [
            {"role": "user", "content": [{"text": "Hi"}]},
            {"role": "assistant", "content": [{"text": "Ok"}]},
            {"role": "user", "content": [{"text": "Are you there?"}]},
        ]

        await answer(llm, conversation, BOB, "Me too")
        await answer(llm, conversation, ALICE, "Who is that?")
        assert bedrock_api[-1]["system"] == [
            {"text": "be brief\n\n" + SEVERAL_SPEAKERS}
        ]
        assert bedrock_api[-1]["messages"][2:] == [
            {"role": "user", "content": [{"text": "Are you there?"}]},
            {"role": "assistant", "content": [{"text": "Ok"}]},
            {"role": "user", "content": [{"text": "Bob: Me too"}]},
            {"role": "assistant", "content": [{"text": "Ok"}]},
            {"role": "user", "content": [{"text": "Alice: Who is that?"}]},
        ]


def _has_aws_creds() -> bool:
    return any(
        os.environ.get(k)
        for k in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
    )


@pytest.mark.integration
class TestBedrockLLMIntegration:
    @pytest.fixture
    async def llm(self):
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Bedrock LLM tests")
        llm = BedrockLLM(model="amazon.nova-lite-v1:0", region_name="us-east-1")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        yield llm
        await llm.close()

    async def test_simple_response(self, llm: BedrockLLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Explain quantum computing in 1 paragraph")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: BedrockLLM):
        await collect_simple_response(
            llm.simple_response("There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_instruction_following(self):
        if not _has_aws_creds():
            pytest.skip("AWS credentials not set – skipping Bedrock LLM tests")
        llm = BedrockLLM(model="qwen.qwen3-32b-v1:0", region_name="us-east-1")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        llm.set_instructions("only reply in 2 letter country shortcuts")
        try:
            _, final = await collect_simple_response(
                llm.simple_response(
                    "Which country is rainy, protected from water with dikes and below sea level?"
                )
            )
            assert "nl" in final.text.lower()
        finally:
            await llm.close()

    async def test_function_calling_live_roundtrip(self, llm: BedrockLLM):
        calls: list[str] = []

        @llm.register_function(
            description="Probe tool that records invocation and returns a marker string"
        )
        async def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "You MUST call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string and nothing else."
        )

        _, final = await collect_simple_response(llm.simple_response(prompt))

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert isinstance(final.text, str)
        assert "probe_ok:pong" in final.text
