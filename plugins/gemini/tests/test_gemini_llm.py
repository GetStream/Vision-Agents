import json
import os
from typing import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dotenv import load_dotenv
from google.genai import Client, types
from google.genai.client import AsyncClient
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS
from vision_agents.plugins.gemini import tools
from vision_agents.plugins.gemini.gemini_llm import (
    GeminiLLM,
)
from vision_agents.plugins.gemini.utils import convert_tools_to_provider_format
from vision_agents.testing import answer, collect_simple_response

load_dotenv()

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


@pytest.fixture
async def gemini_api() -> AsyncIterator[tuple[AsyncClient, list[dict]]]:
    """A real server standing in for Gemini: a client of it, and every request body sent.

    Every answer is one streamed "Ok": the tests are about requests.
    """
    bodies: list[dict] = []
    ok = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Ok"}]}}]}

    async def stream(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(f"data: {json.dumps(ok)}\n\n".encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1beta/models/{call}", stream)
    server = TestServer(app)
    await server.start_server()
    http_options = types.HttpOptions(base_url=str(server.make_url("")))
    yield Client(api_key="test", http_options=http_options).aio, bodies
    await server.close()


@pytest.fixture
async def llm():
    llm = GeminiLLM()
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


class TestGeminiLLM:
    async def test_the_model_is_told_once_a_second_person_has_spoken(
        self, gemini_api: tuple[AsyncClient, list[dict]]
    ):
        client, bodies = gemini_api
        conversation = InMemoryConversation("", [])
        llm = GeminiLLM(model="test", client=client)
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")

        for participant, text in ((ALICE, "Hi"), (ALICE, "Hello?"), (BOB, "Me too")):
            await answer(llm, conversation, participant, text)

        told = [body["systemInstruction"]["parts"][0]["text"] for body in bodies]
        # One person: what the base sha (9653cb696) sent; the chat keeps the instructions.
        assert told == ["be brief", "be brief", "be brief\n\n" + SEVERAL_SPEAKERS]
        assert bodies[-1]["contents"][-1]["parts"] == [{"text": "Bob: Me too"}]

    async def test_each_request_carries_the_built_in_tools_once(
        self, gemini_api: tuple[AsyncClient, list[dict]]
    ):
        client, bodies = gemini_api
        conversation = InMemoryConversation("", [])
        llm = GeminiLLM(
            model="test", client=client, temperature=0.5, tools=[tools.GoogleSearch()]
        )
        llm.set_conversation(conversation)
        llm.set_instructions("be brief")

        turns = ((ALICE, "Hi"), (ALICE, "Hello?"), (BOB, "Me too"), (ALICE, "And"))
        for participant, text in turns:
            await answer(llm, conversation, participant, text)

        assert [len(body["tools"]) for body in bodies] == [1, 1, 1, 1]

    def test_message(self):
        messages = GeminiLLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        advanced = ["say hi"]
        messages2 = GeminiLLM._normalize_message(advanced)
        assert messages2[0].original is not None

    async def test_convert_tools_routes_mcp_schema_to_parameters_json_schema(self):
        tools = [
            {
                "name": "search_docs",
                "description": "Search knowledge base",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                    "$schema": "http://json-schema.org/draft-07/schema#",
                },
            }
        ]

        result = convert_tools_to_provider_format(tools)

        decl = result[0]["function_declarations"][0]
        assert "parameters" not in decl
        schema = decl["parameters_json_schema"]
        assert "$schema" not in schema
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"

    async def test_convert_tools_strips_nested_schema_meta(self):
        tools = [
            {
                "name": "nested",
                "description": "",
                "parameters_schema": {
                    "type": "object",
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "$schema": "http://json-schema.org/draft-07/schema#",
                        }
                    },
                },
            }
        ]

        schema = convert_tools_to_provider_format(tools)[0]["function_declarations"][0][
            "parameters_json_schema"
        ]
        assert "$schema" not in schema
        assert "$schema" not in schema["properties"]["inner"]


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set; skipping live integration test",
)
class TestGeminiLLMIntegration:
    async def test_duplicate_tool_calls_in_followup_do_not_send_empty_parts(
        self, llm: GeminiLLM
    ):
        """If the model echoes the same function call after a tool result,
        `_dedup_and_execute` filters it out and the loop ends up calling
        `chat.send_message_stream(parts=[], ...)`, which google-genai rejects
        with `ValueError('content parts are required.')`.
        """
        # The bug is model-agnostic. We pick a non-"gemini-3" name only to skip
        # the gemini-3 history-cleanup branch, which would require the fake chat
        # to implement get_history() and client.chats.create().
        llm.model = "gemini-2.5-pro"

        @llm.register_function(description="probe")
        async def probe(ping: str) -> str:
            return f"ok:{ping}"

        def _function_call_chunk() -> types.GenerateContentResponse:
            part = types.Part(
                function_call=types.FunctionCall(name="probe", args={"ping": "pong"})
            )
            return types.GenerateContentResponse(
                candidates=[
                    types.Candidate(content=types.Content(role="model", parts=[part]))
                ]
            )

        class _FakeChat:
            def __init__(self) -> None:
                self.received: list[tuple[tuple, dict]] = []

            async def send_message_stream(self, *args: object, **kwargs: object):
                message = args[0] if args else kwargs.get("message")
                # mirror google-genai's t_parts() guard
                if isinstance(message, list) and not message:
                    raise ValueError("content parts are required.")
                self.received.append((args, kwargs))

                async def _iter():
                    yield _function_call_chunk()

                return _iter()

        fake = _FakeChat()
        llm.chat = fake

        try:
            async for _ in llm.simple_response("call probe"):
                pass
        except ValueError as exc:
            if "content parts are required" in str(exc):
                pytest.fail(f"empty-parts bug regressed: {exc}")
            raise

        # Initial text turn + exactly one follow-up with the tool result.
        # A third call would mean we sent parts=[] (the bug).
        assert len(fake.received) == 2
        follow_up_args, _ = fake.received[1]
        assert follow_up_args and follow_up_args[0], "follow-up parts must be non-empty"

    async def test_simple_response(self, llm: GeminiLLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Greet the user")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: GeminiLLM):
        await collect_simple_response(
            llm.simple_response(text="There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response(text="How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_gemini_function_calling_live_roundtrip(self, llm: GeminiLLM):
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

    async def test_instruction_following(self):
        llm = GeminiLLM()
        llm.set_conversation(InMemoryConversation("be friendly", []))

        llm.set_instructions("only reply in 2 letter country shortcuts")

        _, final = await collect_simple_response(
            llm.simple_response(
                text="Which country is rainy, protected from water with dikes and below sea level?",
            )
        )
        assert "nl" in final.text.lower()
        await llm.close()
