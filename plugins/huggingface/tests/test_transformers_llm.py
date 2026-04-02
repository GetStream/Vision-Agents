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
    extract_tool_calls_from_text,
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
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>'
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments_json"] == {"city": "SF"}
        assert calls[0]["id"]

    async def test_generic_json_format(self):
        text = 'Sure: {"name": "get_weather", "arguments": {"city": "NY"}}'
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"

    async def test_nested_arguments(self):
        text = (
            '{"name": "search", "arguments": {"filters": {"owner": "me", "stars": 5}}}'
        )
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments_json"] == {"filters": {"owner": "me", "stars": 5}}

    async def test_hermes_nested_arguments(self):
        text = '<tool_call>{"name": "search", "arguments": {"filters": {"a": {"b": 1}}}}</tool_call>'
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["arguments_json"] == {"filters": {"a": {"b": 1}}}

    async def test_no_tool_calls_in_plain_text(self):
        assert extract_tool_calls_from_text("Hello! How can I help?") == []
        assert (
            extract_tool_calls_from_text('<tool_call>{"name": not json}</tool_call>')
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
            tool_calls, [{"role": "user", "content": "weather?"}], {}
        )

        assert calls_received == ["SF"]
        assert result.text == "Hello there!"

    async def test_tool_call_events_only_emitted_for_final_answer(self, conversation):
        """Tool call markup must never appear in events; only the final
        natural-language answer should produce chunk + completed events."""
        tool_call_text = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "SF"}}</tool_call>'
        )
        final_answer = "It is sunny and 72F in San Francisco."

        decode_outputs = iter([tool_call_text, final_answer])
        tokenizer = _make_mock_tokenizer()
        tokenizer.decode.side_effect = lambda *a, **kw: next(decode_outputs)

        llm = TransformersLLM(model="test-model")
        llm.set_conversation(conversation)
        llm.on_warmed_up(
            ModelResources(
                model=_make_mock_model(),
                tokenizer=tokenizer,
                device=torch.device("cpu"),
            )
        )

        tools_called: list[str] = []

        @llm.register_function("get_weather", description="Get weather")
        async def get_weather(city: str) -> str:
            tools_called.append(city)
            return "Sunny, 72F"

        events_received: list[
            LLMRequestStartedEvent | LLMResponseChunkEvent | LLMResponseCompletedEvent
        ] = []

        @llm.events.subscribe
        async def listen(
            event: LLMRequestStartedEvent
            | LLMResponseChunkEvent
            | LLMResponseCompletedEvent,
        ):
            events_received.append(event)

        result = await llm.create_response(
            messages=[{"role": "user", "content": "weather in SF?"}],
            stream=True,
        )
        await llm.events.wait(1)

        assert tools_called == ["SF"]
        assert result.text == final_answer

        chunk_events = [
            e for e in events_received if isinstance(e, LLMResponseChunkEvent)
        ]
        completed_events = [
            e for e in events_received if isinstance(e, LLMResponseCompletedEvent)
        ]

        assert len(chunk_events) == 1
        assert chunk_events[0].delta == final_answer

        assert len(completed_events) == 1
        assert completed_events[0].text == final_answer

        for evt in chunk_events + completed_events:
            assert "<tool_call>" not in (
                evt.delta if hasattr(evt, "delta") else evt.text
            )

    async def test_multi_round_tool_calls_no_event_leakage(self, conversation):
        """Multiple rounds of tool calls must not leak intermediate text
        into events.  Only the final answer should produce events."""
        round1_text = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "SF"}}</tool_call>'
        )
        round2_text = (
            '<tool_call>{"name": "get_time", '
            '"arguments": {"timezone": "PST"}}</tool_call>'
        )
        final_answer = "It is 2:30 PM and sunny in SF."

        decode_outputs = iter([round1_text, round2_text, final_answer])
        tokenizer = _make_mock_tokenizer()
        tokenizer.decode.side_effect = lambda *a, **kw: next(decode_outputs)

        llm = TransformersLLM(model="test-model")
        llm.set_conversation(conversation)
        llm.on_warmed_up(
            ModelResources(
                model=_make_mock_model(),
                tokenizer=tokenizer,
                device=torch.device("cpu"),
            )
        )

        tools_called: list[str] = []

        @llm.register_function("get_weather", description="Get weather")
        async def get_weather(city: str) -> str:
            tools_called.append(f"weather:{city}")
            return "Sunny, 72F"

        @llm.register_function("get_time", description="Get time")
        async def get_time(timezone: str) -> str:
            tools_called.append(f"time:{timezone}")
            return "14:30"

        events_received: list[LLMResponseChunkEvent | LLMResponseCompletedEvent] = []

        @llm.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent,
        ):
            events_received.append(event)

        result = await llm.create_response(
            messages=[{"role": "user", "content": "weather and time?"}],
            stream=True,
        )
        await llm.events.wait(1)

        assert tools_called == ["weather:SF", "time:PST"]
        assert result.text == final_answer

        chunk_events = [
            e for e in events_received if isinstance(e, LLMResponseChunkEvent)
        ]
        completed_events = [
            e for e in events_received if isinstance(e, LLMResponseCompletedEvent)
        ]

        assert len(chunk_events) == 1
        assert chunk_events[0].delta == final_answer
        assert len(completed_events) == 1
        assert completed_events[0].text == final_answer


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
