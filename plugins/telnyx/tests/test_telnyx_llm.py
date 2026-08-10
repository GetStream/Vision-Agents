"""Tests for the Telnyx LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.telnyx import LLM
from vision_agents.testing import collect_simple_response

load_dotenv()


class TestTelnyxLLM:
    """Unit tests for Telnyx LLM configuration."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TELNYX_API_KEY"):
            LLM()

    async def test_default_model(self):
        llm = LLM(api_key="KEY_test")
        assert llm.model == "meta-llama/Llama-3.3-70B-Instruct"

    async def test_custom_model(self):
        llm = LLM(api_key="KEY_test", model="openai/gpt-4o")
        assert llm.model == "openai/gpt-4o"

    async def test_provider_name(self):
        llm = LLM(api_key="KEY_test")
        assert llm.provider_name == "telnyx"

    async def test_explicit_client_skips_api_key_requirement(self, monkeypatch):
        monkeypatch.delenv("TELNYX_API_KEY", raising=False)
        client = AsyncOpenAI(api_key="KEY_injected", base_url="https://example.invalid")

        assert LLM(client=client).model == "meta-llama/Llama-3.3-70B-Instruct"


@pytest.mark.skipif(not os.getenv("TELNYX_API_KEY"), reason="TELNYX_API_KEY not set")
@pytest.mark.integration
class TestTelnyxLLMIntegration:
    """Integration tests hitting the real Telnyx Inference endpoint."""

    @pytest.fixture
    async def llm(self):
        llm = LLM()
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    async def test_simple_response(self, llm):
        deltas, final = await collect_simple_response(
            llm.simple_response("Greet the user in English")
        )
        assert final.text
        assert deltas

    async def test_streaming_chunks(self, llm):
        deltas, final = await collect_simple_response(
            llm.simple_response("List the first 3 prime numbers, separated by commas.")
        )
        assert final.text
        assert len(deltas) > 0, f"No chunks emitted. Response text: {final.text!r}"

    async def test_function_calling(self, llm):
        calls: list[str] = []

        @llm.register_function(description="Probe tool that records invocation")
        async def probe_tool(ping: str) -> str:
            calls.append(ping)
            return f"probe_ok:{ping}"

        prompt = (
            "Call the tool named 'probe_tool' with the parameter ping='pong' now. "
            "After receiving the tool result, reply by returning ONLY the tool result string."
        )
        _, final = await collect_simple_response(llm.simple_response(prompt))

        assert "pong" in calls, (
            f"probe_tool was not invoked with ping='pong' (got calls={calls})"
        )
        assert "probe_ok:pong" in final.text, (
            f"Expected 'probe_ok:pong', got: {final.text}"
        )
