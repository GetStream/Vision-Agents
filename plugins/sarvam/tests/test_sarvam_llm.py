"""Tests for the Sarvam LLM plugin."""

import os

import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import LLMResponseChunkEvent
from vision_agents.plugins.sarvam import LLM
from vision_agents.plugins.sarvam.llm import _ThinkTagFilter

load_dotenv()


class TestThinkTagFilter:
    """Unit tests for the streaming think-tag stripper."""

    def test_no_think_tags(self):
        f = _ThinkTagFilter()
        assert f.feed("Hello world") == "Hello world"

    def test_complete_think_block(self):
        f = _ThinkTagFilter()
        assert f.feed("<think>reasoning</think>answer") == "answer"

    def test_think_block_split_across_feeds(self):
        f = _ThinkTagFilter()
        assert f.feed("<think>start of ") == ""
        assert f.feed("reasoning</think>") == ""
        assert f.feed("the answer") == "the answer"

    def test_closing_tag_split_across_feeds(self):
        f = _ThinkTagFilter()
        assert f.feed("<think>reasoning") == ""
        assert f.feed("</") == ""
        assert f.feed("think") == ""
        assert f.feed(">\n\nthe answer") == "\n\nthe answer"

    def test_partial_open_tag_across_feeds(self):
        f = _ThinkTagFilter()
        assert f.feed("hello <thi") == "hello "
        assert f.feed("nk>thinking</think>world") == "world"

    def test_text_before_and_after(self):
        f = _ThinkTagFilter()
        assert f.feed("before<think>inner</think>after") == "beforeafter"

    def test_flush_strips_remaining(self):
        f = _ThinkTagFilter()
        raw = "<think>let me think about this</think>The answer is 42."
        f.feed(raw)
        assert f.flush(raw) == "The answer is 42."

    def test_no_false_positive_on_angle_bracket(self):
        f = _ThinkTagFilter()
        assert f.feed("x < y and y > z") == "x < y and y > z"


class TestSarvamLLM:
    """Unit tests for Sarvam LLM configuration."""

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="SARVAM_API_KEY"):
            LLM()

    async def test_default_model(self):
        llm = LLM(api_key="sk_test")
        assert llm.model == "sarvam-m"

    async def test_custom_model(self):
        llm = LLM(api_key="sk_test", model="sarvam-30b")
        assert llm.model == "sarvam-30b"

    async def test_base_url_points_to_sarvam(self):
        llm = LLM(api_key="sk_test")
        assert str(llm._client.base_url).startswith("https://api.sarvam.ai")

    async def test_subscription_key_header_injected(self):
        llm = LLM(api_key="sk_test")
        headers = llm._client._custom_headers
        assert headers.get("api-subscription-key") == "sk_test"


@pytest.mark.skipif(not os.getenv("SARVAM_API_KEY"), reason="SARVAM_API_KEY not set")
@pytest.mark.integration
class TestSarvamLLMIntegration:
    """Integration tests hitting the real Sarvam Chat Completions endpoint."""

    @pytest.fixture
    async def llm(self):
        llm = LLM(model="sarvam-m")
        llm.set_conversation(InMemoryConversation("be friendly", []))
        return llm

    async def test_simple_response(self, llm):
        response = await llm.simple_response("Greet the user in English")
        assert response.text
        assert response.exception is None

    async def test_streaming_chunks(self, llm):
        chunks: list[str] = []

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            chunks.append(event.delta)

        response = await llm.simple_response(
            "List the first 3 prime numbers, separated by commas."
        )
        await llm.events.wait()
        assert response.text
        assert len(chunks) > 0, f"No chunks emitted. Response text: {response.text!r}"

    async def test_think_tags_stripped_from_response(self, llm):
        response = await llm.simple_response("What is 2+2? Answer in one word.")
        assert "<think>" not in response.text
        assert "</think>" not in response.text

    async def test_think_tags_stripped_from_chunks(self, llm):
        chunks: list[str] = []

        @llm.events.subscribe
        async def on_chunk(event: LLMResponseChunkEvent):
            chunks.append(event.delta)

        await llm.simple_response("What is 2+2? Answer in one word.")
        await llm.events.wait()
        full = "".join(chunks)
        assert "<think>" not in full
        assert "</think>" not in full
