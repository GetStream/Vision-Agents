import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.plugins.gemini.gemini_llm import (
    GeminiLLM,
)

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture
async def llm():
    llm = GeminiLLM()
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


class TestGeminiLLM:
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

    @pytest.fixture
    async def llm(self):
        llm = GeminiLLM()
        llm.set_conversation(InMemoryConversation("be friendly", []))
        yield llm
        await llm.close()

    @pytest.mark.integration
    async def test_simple(self, llm: GeminiLLM):
        response = await llm.simple_response("Greet the user")
        assert response.text

    @pytest.mark.integration
    async def test_native_api(self, llm: GeminiLLM):
        response = await llm.send_message(message="say hi")

        # Assertions
        assert response.text
        assert hasattr(response.original, "text")  # Gemini response has text attribute

    @pytest.mark.integration
    async def test_stream(self, llm: GeminiLLM):
        streaming_works = False

        @llm.events.subscribe
        async def passed(event: LLMResponseChunkEvent):
            nonlocal streaming_works
            streaming_works = True

        await llm.simple_response("Greet the user")

        # Wait for all events in queue to be processed
        await llm.events.wait()

        assert streaming_works

    async def test_convert_tools_routes_mcp_schema_to_parameters_json_schema(
        self, llm: GeminiLLM
    ):
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

        result = llm._convert_tools_to_provider_format(tools)

        decl = result[0]["function_declarations"][0]
        assert "parameters" not in decl
        schema = decl["parameters_json_schema"]
        assert "$schema" not in schema
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["type"] == "string"

    async def test_convert_tools_strips_nested_schema_meta(self, llm: GeminiLLM):
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

        schema = llm._convert_tools_to_provider_format(tools)[0][
            "function_declarations"
        ][0]["parameters_json_schema"]
        assert "$schema" not in schema
        assert "$schema" not in schema["properties"]["inner"]


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
    reason="GOOGLE_API_KEY or GEMINI_API_KEY not set; skipping live integration test",
)
class TestGeminiLLMIntegration:
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
