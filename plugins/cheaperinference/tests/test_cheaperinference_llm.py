"""Tests for Cheaper Inference LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.cheaperinference import LLM

from vision_agents.testing import collect_simple_response

load_dotenv()


@pytest.fixture()
async def llm_factory():
    """Fixture for Cheaper Inference LLM with conversation."""

    def factory(
        model: str = "gpt-5.4-mini",
        instructions: str = "be friendly",
        max_tokens: int | None = 128,
    ) -> LLM:
        llm = LLM(
            model=model,
            max_tokens=max_tokens,
            api_key=os.environ.get("CHEAPER_INFERENCE_API_KEY") or "test",
        )
        llm.set_conversation(InMemoryConversation(instructions, []))
        return llm

    return factory


class TestCheaperInferenceLLM:
    """Test suite for Cheaper Inference LLM class."""

    def test_default_model(self):
        """Default model should be gpt-5.4-mini."""
        llm = LLM(api_key="test")
        assert llm.model == "gpt-5.4-mini"

    def test_provider_name(self):
        """provider_name should be 'cheaperinference'."""
        llm = LLM(api_key="test")
        assert llm.provider_name == "cheaperinference"

    def test_custom_model(self):
        """Custom model should override the default."""
        llm = LLM(model="claude-sonnet-5", api_key="test")
        assert llm.model == "claude-sonnet-5"

    def test_default_base_url(self, monkeypatch):
        """The client should target the Cheaper Inference endpoint by default."""
        monkeypatch.delenv("CHEAPER_INFERENCE_BASE_URL", raising=False)
        llm = LLM(api_key="test")
        assert str(llm._client.base_url) == "https://api.cheaperinference.com/v1/"

    def test_env_configuration(self, monkeypatch):
        """API key and base URL should load from environment variables."""
        monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", "env-key")
        monkeypatch.setenv("CHEAPER_INFERENCE_BASE_URL", "https://example.test/v1")
        llm = LLM()
        assert llm._client.api_key == "env-key"
        assert str(llm._client.base_url) == "https://example.test/v1/"

    async def test_build_model_request_includes_instructions(self, llm_factory):
        """System instructions must be the first message in the request."""
        llm = llm_factory()
        llm.set_instructions("be terse")
        messages = await llm._build_model_request()
        assert messages and messages[0]["role"] == "system"
        assert messages[0]["content"] == "be terse"


@pytest.mark.skipif(
    not os.getenv("CHEAPER_INFERENCE_API_KEY"),
    reason="CHEAPER_INFERENCE_API_KEY not set",
)
@pytest.mark.integration
class TestCheaperInferenceLLMIntegration:
    async def test_simple_response(self, llm_factory):
        """Test simple response yields deltas and a final."""
        llm = llm_factory()
        deltas, final = await collect_simple_response(
            llm.simple_response("Greet the user")
        )
        assert deltas, "Streaming should yield deltas"
        assert final.text

    async def test_memory(self, llm_factory):
        """Test conversation memory using simple_response."""
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
        """Test function calling with the Cheaper Inference LLM."""
        llm = llm_factory(max_tokens=512)
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
