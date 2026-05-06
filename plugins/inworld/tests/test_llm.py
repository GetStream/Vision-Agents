"""Tests for the Inworld LLM/VLM router plugin."""

import os

import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins import inworld
from vision_agents.plugins.inworld.llm import INWORLD_BASE_URL
from vision_agents.testing import collect_simple_response

load_dotenv()


class TestInworldLLM:
    """Unit tests for InworldLLM construction and routing config."""

    @pytest.fixture
    async def llm(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "test-key-unit")
        llm = inworld.LLM()
        try:
            yield llm
        finally:
            await llm.close()

    async def test_default_base_url_and_api_key_env(self, llm):
        assert str(llm._client.base_url).rstrip("/") == INWORLD_BASE_URL.rstrip("/")
        assert llm._client.api_key == "test-key-unit"
        assert llm.model == "auto"

    async def test_extra_body_construction(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        llm = inworld.LLM(
            model="openai/gpt-4o-mini",
            fallback_models=[
                "anthropic/claude-sonnet-4-5",
                "google-ai-studio/gemini-2.5-flash",
            ],
            ignore_models=["openai/gpt-3.5-turbo"],
            sort_by=["latency", "price"],
            ttft_timeout="500ms",
            metadata={"tier": "premium"},
            web_search=True,
            web_search_options={"max_results": 3},
            extra_body={"custom_field": 42},
        )
        try:
            assert llm._extra_body == {
                "models": [
                    "anthropic/claude-sonnet-4-5",
                    "google-ai-studio/gemini-2.5-flash",
                ],
                "ignore": ["openai/gpt-3.5-turbo"],
                "sort": ["latency", "price"],
                "fallback": {"ttft_timeout": "500ms"},
                "metadata": {"tier": "premium"},
                "web_search": True,
                "web_search_options": {"max_results": 3},
                "custom_field": 42,
            }
            assert llm._extra_request_kwargs() == {"extra_body": llm._extra_body}
        finally:
            await llm.close()

    async def test_extra_request_kwargs_empty_when_no_routing(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        llm = inworld.LLM(model="openai/gpt-4o-mini")
        try:
            assert llm._extra_body == {}
            assert llm._extra_request_kwargs() == {}
        finally:
            await llm.close()

    async def test_explicit_client_wins_over_api_key(self):
        custom = AsyncOpenAI(api_key="sentinel", base_url="https://example.invalid/v1")
        try:
            llm = inworld.LLM(api_key="ignored", client=custom)
            assert llm._client is custom
            assert llm._client.api_key == "sentinel"
        finally:
            await custom.close()

    async def test_compression_applied_to_system_message_only(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        llm = inworld.LLM(compression_aggressiveness=0.5)
        try:
            llm.set_instructions("be brief")
            llm.set_conversation(InMemoryConversation("be brief", []))
            await llm._conversation.send_message(role="user", user_id="u", content="hi")

            messages = await llm._build_model_request()

            compressed = [m for m in messages if "compression" in m]
            assert len(compressed) == 1
            assert compressed[0]["role"] == "system"
            assert compressed[0]["compression"] == {"aggressiveness": 0.5}
        finally:
            await llm.close()

    async def test_no_compression_when_kwarg_unset(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        llm = inworld.LLM()
        try:
            llm.set_instructions("be brief")
            llm.set_conversation(InMemoryConversation("be brief", []))
            messages = await llm._build_model_request()
            assert all("compression" not in m for m in messages)
        finally:
            await llm.close()


class TestInworldVLM:
    """Unit tests for InworldVLM — same routing wiring as LLM."""

    async def test_default_base_url_and_extra_body_shape(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        vlm = inworld.VLM(
            model="openai/gpt-4o",
            sort_by=["latency"],
            ttft_timeout="500ms",
            fallback_models=["google-ai-studio/gemini-2.5-flash"],
        )
        try:
            assert str(vlm._client.base_url).rstrip("/") == INWORLD_BASE_URL.rstrip("/")
            assert vlm._extra_body == {
                "sort": ["latency"],
                "fallback": {"ttft_timeout": "500ms"},
                "models": ["google-ai-studio/gemini-2.5-flash"],
            }
            assert vlm._extra_request_kwargs() == {"extra_body": vlm._extra_body}
        finally:
            await vlm.close()

    async def test_frame_kwargs_pass_through_to_parent(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        vlm = inworld.VLM(
            model="openai/gpt-4o",
            fps=2,
            frame_buffer_seconds=3,
            frame_width=512,
            frame_height=384,
        )
        try:
            assert vlm._fps == 2
            assert vlm._frame_buffer.maxlen == 6
            assert vlm._frame_width == 512
            assert vlm._frame_height == 384
        finally:
            await vlm.close()

    async def test_compression_applied_to_system_message(self, monkeypatch):
        monkeypatch.setenv("INWORLD_API_KEY", "k")
        vlm = inworld.VLM(model="openai/gpt-4o", compression_aggressiveness=0.7)
        try:
            vlm.set_instructions("describe what you see")
            vlm.set_conversation(InMemoryConversation("describe what you see", []))

            messages = await vlm._build_model_request()

            compressed = [m for m in messages if "compression" in m]
            assert len(compressed) == 1
            assert compressed[0]["role"] == "system"
            assert compressed[0]["compression"] == {"aggressiveness": 0.7}
        finally:
            await vlm.close()


@pytest.mark.skipif(not os.getenv("INWORLD_API_KEY"), reason="INWORLD_API_KEY not set")
@pytest.mark.integration
class TestInworldLLMIntegration:
    """End-to-end tests against the live Inworld router. Require ``INWORLD_API_KEY``."""

    @pytest.fixture
    async def llm(self):
        llm = inworld.LLM(
            model="auto",
            sort_by=["latency"],
            fallback_models=["openai/gpt-4o-mini"],
        )
        llm.set_conversation(InMemoryConversation("be brief", []))
        try:
            yield llm
        finally:
            await llm.close()

    async def test_simple_response_streams_text(self, llm):
        deltas, final = await collect_simple_response(
            llm.simple_response("Say hello in one short sentence.")
        )
        assert deltas, "router should stream at least one delta"
        assert final.text, "final response text should not be empty"
        # Performance regression guard: verify we got per-chunk streaming, not a
        # single buffered response. ttft_ms < total_latency_ms by a clear margin.
        first_ttft = deltas[0].time_to_first_token_ms
        assert first_ttft is not None and first_ttft < 5000, (
            f"first-token latency {first_ttft}ms exceeds 5s — streaming may have "
            "fallen back to non-streaming, or auto+latency routing is degraded"
        )

    async def test_tool_calling_round_trip(self, llm):
        @llm.register_function(description="Get the weather for a city")
        async def get_weather(city: str) -> str:
            return f"It is sunny and 22°C in {city}."

        _, final = await collect_simple_response(
            llm.simple_response("What's the weather in Berlin?")
        )
        assert "berlin" in final.text.lower() or "sunny" in final.text.lower(), (
            f"expected tool result to influence final answer; got: {final.text!r}"
        )
