"""Tests for TransformersLLM - local text LLM inference."""

import os
from unittest.mock import MagicMock

import pytest
import torch
from conftest import skip_blockbuster
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.testing import collect_simple_response
from vision_agents.core.llm.llm import LLMResponseFinal
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
        """Streaming response yields deltas and a final."""
        await conversation.send_message(
            role="user", user_id="user1", content="prior message"
        )

        deltas, final = await collect_simple_response(llm.simple_response(text="hello"))

        assert final.text == "Hello there!"
        assert "".join(d.delta or "" for d in deltas) == "Hello there!"
        assert all(d.delta for d in deltas), "deltas must not be empty"

        # Verify messages were built from conversation
        tokenizer = llm._resources.tokenizer
        messages = tokenizer.apply_chat_template.call_args.args[0]
        assert any(m.get("content") == "hello" for m in messages)

    async def test_non_streaming_response(self, llm):
        messages = [{"role": "user", "content": "test"}]
        _, final = await collect_simple_response(
            llm.create_response(messages=messages, stream=False)
        )
        assert final.text == "Hello there!"

    async def test_generation_error(self, llm):
        llm._resources.model.generate.side_effect = RuntimeError("OOM")

        messages = [{"role": "user", "content": "test"}]
        deltas, final = await collect_simple_response(
            llm.create_response(messages=messages, stream=False)
        )

        assert final.text == ""
        assert deltas == []

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

        _, final = await collect_simple_response(
            llm.create_response(
                messages=[{"role": "user", "content": "test"}], stream=False
            )
        )

        assert call_count == 2
        assert final.text == "Hello there!"


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
    async def test_tool_calls_execute_and_generate_followup(self, conversation):
        """Tool calls extracted from model output are executed and a follow-up
        round produces the final answer."""
        tool_call_text = (
            '<tool_call>{"name": "get_weather", '
            '"arguments": {"city": "SF"}}</tool_call>'
        )
        final_answer = "Sunny, 72F in SF."

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

        calls_received: list[str] = []

        @llm.register_function("get_weather", description="Get weather")
        async def get_weather(city: str) -> str:
            calls_received.append(city)
            return "Sunny, 72F"

        _, final = await collect_simple_response(
            llm.create_response(
                messages=[{"role": "user", "content": "weather?"}], stream=True
            )
        )

        assert calls_received == ["SF"]
        assert final.text == final_answer

    async def test_tool_call_markup_not_yielded_as_deltas(self, conversation):
        """Tool call markup must never appear in yielded deltas; only the
        final natural-language answer is yielded."""
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

        deltas, final = await collect_simple_response(
            llm.create_response(
                messages=[{"role": "user", "content": "weather in SF?"}],
                stream=True,
            )
        )

        assert tools_called == ["SF"]
        assert final.text == final_answer
        assert len(deltas) == 1
        assert deltas[0].delta == final_answer
        assert "<tool_call>" not in final.text
        for d in deltas:
            assert "<tool_call>" not in (d.delta or "")

    async def test_multi_round_tool_calls_no_delta_leakage(self, conversation):
        """Multiple rounds of tool calls must not leak intermediate text into
        yielded deltas. Only the final answer is yielded."""
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

        deltas, final = await collect_simple_response(
            llm.create_response(
                messages=[{"role": "user", "content": "weather and time?"}],
                stream=True,
            )
        )

        assert tools_called == ["weather:SF", "time:PST"]
        assert final.text == final_answer
        assert len(deltas) == 1
        assert deltas[0].delta == final_answer


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
        await llm.warmup()

        deltas, final = await collect_simple_response(
            llm.simple_response(text="Greet the user")
        )
        assert len(deltas) > 0
        assert final.text

        llm.unload()

    async def test_interrupt_stops_generation(self):
        """Calling ``interrupt()`` mid-generation stops ``model.generate``
        within ≤1 token, even though the asyncio task is still iterating."""
        model_id = os.getenv("TRANSFORMERS_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

        llm = TransformersLLM(model=model_id, max_new_tokens=500)
        conversation = InMemoryConversation("", [])
        llm.set_conversation(conversation)
        await llm.warmup()

        deltas = []
        final = None
        async for item in llm.simple_response(
            text="Count from 1 to 1000 spelling out every number in full English words"
        ):
            if isinstance(item, LLMResponseFinal):
                final = item
            else:
                deltas.append(item)
                if len(deltas) == 1:
                    await llm.interrupt()

        assert final is not None
        # Without interrupt the response would be hundreds of tokens. With
        # interrupt fired after the first delta, the rest of the run produces
        # only a handful more tokens before ``model.generate`` exits.
        assert len(deltas) < 20

        llm.unload()
