"""Tests for TransformersLLM - local text LLM inference."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from conftest import skip_blockbuster
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.plugins.huggingface.events import LLMErrorEvent
from vision_agents.plugins.huggingface.transformers_llm import (
    ModelResources,
    TransformersLLM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(decoded_text: str = "Hello there!") -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    tokenizer.decode.return_value = decoded_text
    return tokenizer


def _make_mock_model(output_ids: list[int] | None = None) -> MagicMock:
    model = MagicMock()
    if output_ids is None:
        output_ids = [1, 2, 3, 10, 11, 12]

    ids = output_ids

    def _generate_side_effect(**kwargs):
        streamer = kwargs.get("streamer")
        if streamer:
            streamer.put(torch.tensor(ids[:3]))
            streamer.put(torch.tensor(ids[3:]))
            streamer.end()
        return torch.tensor([ids])

    model.generate.side_effect = _generate_side_effect

    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.return_value = iter([param])
    return model


def _make_resources(decoded_text: str = "Hello there!") -> ModelResources:
    return ModelResources(
        model=_make_mock_model(),
        tokenizer=_make_mock_tokenizer(decoded_text),
        device=torch.device("cpu"),
    )


@pytest.fixture()
async def conversation():
    return InMemoryConversation("", [])


@pytest.fixture()
async def llm(conversation):
    llm_ = TransformersLLM(model="test-model")
    llm_.set_conversation(conversation)
    llm_.on_warmed_up(_make_resources())
    return llm_


# ---------------------------------------------------------------------------
# Mocked tests
# ---------------------------------------------------------------------------


@skip_blockbuster
class TestTransformersLLM:
    async def test_simple_response(self, llm, conversation):
        """Streaming response returns text and emits expected events."""
        await conversation.send_message(
            role="user", user_id="user1", content="prior message"
        )

        events_received = []

        @llm.events.subscribe
        async def listen(
            event: LLMRequestStartedEvent
            | LLMResponseChunkEvent
            | LLMResponseCompletedEvent,
        ):
            events_received.append(event)

        response = await llm.simple_response(text="hello")
        await llm.events.wait(1)

        assert response.text == "Hello there!"

        event_types = [e.type for e in events_received]
        assert "plugin.llm_request_started" in event_types
        assert "plugin.llm_response_completed" in event_types

        # Verify messages were built from conversation
        tokenizer = llm._resources.tokenizer
        messages = tokenizer.apply_chat_template.call_args.args[0]
        assert any(m.get("content") == "hello" for m in messages)

    async def test_non_streaming_response(self, llm):
        messages = [{"role": "user", "content": "test"}]
        response = await llm.create_response(messages=messages, stream=False)
        assert response.text == "Hello there!"

    async def test_generation_error(self, llm):
        llm._resources.model.generate.side_effect = RuntimeError("OOM")

        error_events = []

        @llm.events.subscribe
        async def listen(event: LLMErrorEvent):
            error_events.append(event)

        messages = [{"role": "user", "content": "test"}]
        response = await llm.create_response(messages=messages, stream=False)
        await llm.events.wait(1)

        assert response.text == ""
        assert len(error_events) == 1
        assert "OOM" in error_events[0].error_message

    async def test_chat_template_tools_fallback(self, llm):
        """When apply_chat_template fails with tools, retries without."""
        tokenizer = llm._resources.tokenizer
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "tools" in kwargs:
                raise ValueError("Template does not support tools")
            real_ids = torch.tensor([[1, 2, 3]])
            return {"input_ids": real_ids, "attention_mask": torch.ones_like(real_ids)}

        tokenizer.apply_chat_template.side_effect = side_effect

        @llm.register_function(description="A test tool")
        async def test_tool() -> str:
            return "result"

        response = await llm.create_response(
            messages=[{"role": "user", "content": "test"}], stream=False
        )

        assert call_count == 2
        assert response.text == "Hello there!"


class TestToolCallParsing:
    async def test_hermes_format(self):
        llm = TransformersLLM(model="test")
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        calls = llm._extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments_json"] == {"city": "SF"}
        assert calls[0]["id"]

    async def test_generic_json_format(self):
        llm = TransformersLLM(model="test")
        text = 'Sure: {"name": "get_weather", "arguments": {"city": "NY"}}'
        calls = llm._extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    async def test_no_tool_calls_in_plain_text(self):
        llm = TransformersLLM(model="test")
        assert llm._extract_tool_calls_from_text("Hello! How can I help?") == []
        assert (
            llm._extract_tool_calls_from_text(
                '<tool_call>{"name": not json}</tool_call>'
            )
            == []
        )


@skip_blockbuster
class TestToolCallExecution:
    async def test_tool_calls_execute_and_generate_followup(self, llm, conversation):
        calls_received = []

        @llm.register_function("get_weather", description="Get weather")
        async def get_weather(city: str) -> str:
            calls_received.append(city)
            return "Sunny, 72F"

        tool_calls = [
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "get_weather",
                "arguments_json": {"city": "SF"},
            }
        ]

        result = await llm._handle_tool_calls(
            tool_calls, [{"role": "user", "content": "weather?"}], [], {}
        )

        assert calls_received == ["SF"]
        assert result.text == "Hello there!"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_blockbuster
class TestTransformersLLMIntegration:
    async def test_simple_response(self):
        model_id = os.getenv("TRANSFORMERS_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

        llm = TransformersLLM(model=model_id, max_new_tokens=30)
        conversation = InMemoryConversation("", [])
        llm.set_conversation(conversation)

        resources = await llm.on_warmup()
        llm.on_warmed_up(resources)

        response = await llm.simple_response(text="Say hello in one word")
        assert response.text
        assert len(response.text) > 0

        llm.unload()
