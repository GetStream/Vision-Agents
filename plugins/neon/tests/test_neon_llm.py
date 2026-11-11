"""Tests for Neon AI Gateway LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.neon import LLM

from vision_agents.testing import collect_simple_response

load_dotenv()

TEST_BASE_URL = "https://br-test-api.ai.c-2.us-east-2.aws.neon.tech"


@pytest.fixture()
async def llm_factory():
    """Fixture for Neon LLM with conversation."""

    def factory(
        model: str = "gpt-5-mini",
        instructions: str = "be friendly",
        max_tokens: int | None = 512,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("NEON_AI_GATEWAY_TOKEN") or "test",
            base_url=os.environ.get("NEON_AI_GATEWAY_BASE_URL") or TEST_BASE_URL,
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestNeonLLM:
    """Test suite for Neon LLM class."""

    def test_defaults(self):
        llm = LLM(api_key="test", base_url=TEST_BASE_URL)
        assert llm.model == "gpt-5-mini"
        assert llm.provider_name == "neon"

    def test_base_url_points_at_chat_completions(self):
        llm = LLM(api_key="test", base_url=f"{TEST_BASE_URL}/")
        assert str(llm._client.base_url) == f"{TEST_BASE_URL}/v1/"

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("NEON_AI_GATEWAY_TOKEN", "nt_live_test")
        monkeypatch.setenv("NEON_AI_GATEWAY_BASE_URL", TEST_BASE_URL)
        llm = LLM()
        assert llm._client.api_key == "nt_live_test"
        assert str(llm._client.base_url) == f"{TEST_BASE_URL}/v1/"

    def test_missing_token_raises(self, monkeypatch):
        monkeypatch.delenv("NEON_AI_GATEWAY_TOKEN", raising=False)
        with pytest.raises(ValueError, match="NEON_AI_GATEWAY_TOKEN"):
            LLM(base_url=TEST_BASE_URL)

    def test_missing_base_url_raises(self, monkeypatch):
        monkeypatch.delenv("NEON_AI_GATEWAY_BASE_URL", raising=False)
        with pytest.raises(ValueError, match="NEON_AI_GATEWAY_BASE_URL"):
            LLM(api_key="test")


@pytest.mark.skipif(
    not (os.getenv("NEON_AI_GATEWAY_TOKEN") and os.getenv("NEON_AI_GATEWAY_BASE_URL")),
    reason="NEON_AI_GATEWAY_TOKEN / NEON_AI_GATEWAY_BASE_URL not set",
)
@pytest.mark.integration
class TestNeonLLMIntegration:
    async def test_simple_response(self, llm_factory):
        llm = llm_factory()
        deltas, final = await collect_simple_response(
            llm.simple_response("Greet the user")
        )
        assert deltas, "Streaming should yield deltas"
        assert final.text

    async def test_memory(self, llm_factory):
        llm = llm_factory()
        await collect_simple_response(
            llm.simple_response("There are 2 dogs in the room")
        )
        _, final = await collect_simple_response(
            llm.simple_response("How many paws are there in the room?")
        )
        assert "8" in final.text or "eight" in final.text.lower(), (
            f"Expected '8' or 'eight' in response, got: {final.text}"
        )

    async def test_function_calling(self, llm_factory):
        llm = llm_factory()
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

        assert len(calls) >= 1, "probe_tool was not invoked by the model"
        assert "probe_ok:pong" in final.text, (
            f"Expected 'probe_ok:pong', got: {final.text}"
        )
