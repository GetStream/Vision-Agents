import copy
import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation, Message
from vision_agents.core.llm import FunctionRegistry
from vision_agents.plugins.openai.openai_llm import OpenAILLM
from vision_agents.plugins.openai.tool_utils import convert_tools_to_openai_format
from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture
async def llm():
    llm = OpenAILLM(model="gpt-4o")
    llm.set_conversation(InMemoryConversation("be friendly", []))
    yield llm
    await llm.close()


class TestOpenAILLM:
    """Test suite for OpenAILLM class with mocked API calls."""

    def test_message(self):
        messages = OpenAILLM._normalize_message("say hi")
        assert isinstance(messages[0], Message)
        message = messages[0]
        assert message.original is not None
        assert message.content == "say hi"

    def test_advanced_message(self):
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/2023_06_08_Raccoon1.jpg/1599px-2023_06_08_Raccoon1.jpg"

        advanced = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what do you see in this image?"},
                    {"type": "input_image", "image_url": f"{img_url}"},
                ],
            }
        ]
        messages2 = OpenAILLM._normalize_message(advanced)
        assert messages2[0].original is not None


class TestOpenAIToolConversion:
    """Unit tests for OpenAI function tool schema conversion."""

    def test_required_only_schema_uses_strict_mode(self):
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            }
        ]

        converted = convert_tools_to_openai_format(tools)
        parameters = converted[0]["parameters"]

        assert converted[0]["strict"] is True
        assert parameters["additionalProperties"] is False
        assert parameters["required"] == ["location"]

    def test_optional_property_schema_disables_strict_mode(self):
        tools = [
            {
                "name": "check_inventory_tool",
                "description": "Check fake demo inventory for a medication.",
                "parameters_schema": {
                    "type": "object",
                    "properties": {
                        "medication_name": {"type": "string"},
                        "strength": {"type": "string"},
                    },
                    "required": ["medication_name"],
                },
            }
        ]

        converted = convert_tools_to_openai_format(tools)
        parameters = converted[0]["parameters"]

        assert converted[0]["strict"] is False
        assert parameters["required"] == ["medication_name"]
        assert "strength" in parameters["properties"]

    def test_realtime_schema_omits_strict_mode(self):
        tools = [
            {
                "name": "get_weather",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        converted = convert_tools_to_openai_format(tools, for_realtime=True)

        assert "strict" not in converted[0]

    def test_converter_does_not_mutate_original_schema(self):
        tools = [
            {
                "name": "get_weather",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]
        original = copy.deepcopy(tools)

        convert_tools_to_openai_format(tools)

        assert tools == original

    def test_registered_optional_parameter_is_openai_compatible(self):
        registry = FunctionRegistry()

        @registry.register(description="Check fake demo inventory for a medication.")
        async def check_inventory_tool(
            medication_name: str, strength: str | None = None
        ):
            return {"medication_name": medication_name, "strength": strength}

        converted = convert_tools_to_openai_format(registry.get_tool_schemas())
        parameters = converted[0]["parameters"]

        assert converted[0]["strict"] is False
        assert parameters["required"] == ["medication_name"]
        assert "strength" in parameters["properties"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live integration test",
)
class TestOpenAILLMIntegration:
    async def test_stream(self, llm: OpenAILLM):
        deltas, final = await collect_simple_response(
            llm.simple_response("Explain quantum computing in 1 paragraph")
        )
        assert deltas
        assert final.text

    async def test_memory(self, llm: OpenAILLM):
        await collect_simple_response(
            llm.simple_response("There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text

    async def test_openai_function_calling_live_roundtrip(self, llm):
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
