"""Tests for TransformersVLM - local vision-language model inference."""

import fractions
import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from av import VideoFrame
from conftest import skip_blockbuster
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.plugins.huggingface.transformers_vlm import (
    TransformersVLM,
    VLMResources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_processor(decoded_text: str = "A cat on a couch") -> MagicMock:
    processor = MagicMock()

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    processor.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    processor.decode.return_value = decoded_text
    processor.tokenizer = MagicMock()
    processor.tokenizer.pad_token_id = 0
    return processor


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    model.generate.return_value = torch.tensor([[1, 2, 3, 20, 21, 22]])

    param = torch.nn.Parameter(torch.zeros(1))
    model.parameters.return_value = iter([param])
    return model


def _make_resources(decoded_text: str = "A cat on a couch") -> VLMResources:
    return VLMResources(
        model=_make_mock_model(),
        processor=_make_mock_processor(decoded_text),
        device=torch.device("cpu"),
    )


def _random_video_frame() -> VideoFrame:
    array = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    frame = VideoFrame.from_ndarray(array, format="bgr24")
    frame.pts = 0
    frame.time_base = fractions.Fraction(1, 30)
    return frame


@pytest.fixture()
async def conversation():
    return InMemoryConversation("", [])


@pytest.fixture()
async def vlm(conversation):
    vlm_ = TransformersVLM(model="test-vlm")
    vlm_.set_conversation(conversation)
    vlm_.on_warmed_up(_make_resources())
    return vlm_


# ---------------------------------------------------------------------------
# Mocked tests
# ---------------------------------------------------------------------------


@skip_blockbuster
class TestTransformersVLM:
    async def test_simple_response_with_frames(self, vlm, conversation):
        """Response with video frames passes images to processor and emits events."""
        for _ in range(3):
            vlm._frame_buffer.append(_random_video_frame())

        events_received = []

        @vlm.events.subscribe
        async def listen(
            event: VLMInferenceStartEvent
            | VLMInferenceCompletedEvent
            | LLMResponseCompletedEvent,
        ):
            events_received.append(event)

        response = await vlm.simple_response(text="what do you see?")
        await vlm.events.wait(1)

        assert response.text == "A cat on a couch"

        # Verify images were passed to processor
        processor = vlm._resources.processor
        call_args = processor.apply_chat_template.call_args
        assert len(call_args.kwargs["images"]) == 3

        # Verify VLM events
        event_types = [e.type for e in events_received]
        assert "plugin.vlm_inference_start" in event_types
        assert "plugin.vlm_inference_completed" in event_types
        assert "plugin.llm_response_completed" in event_types

    async def test_simple_response_no_frames(self, vlm, conversation):
        """Response works with empty frame buffer (images=None)."""
        response = await vlm.simple_response(text="describe")
        assert response.text == "A cat on a couch"

        processor = vlm._resources.processor
        call_args = processor.apply_chat_template.call_args
        assert call_args.kwargs.get("images") is None

    async def test_generation_error(self, vlm, conversation):
        vlm._resources.model.generate.side_effect = RuntimeError("OOM")

        error_events = []

        @vlm.events.subscribe
        async def listen(event: VLMErrorEvent):
            error_events.append(event)

        response = await vlm.simple_response(text="describe")
        await vlm.events.wait(1)

        assert response.text == ""
        assert len(error_events) == 1
        assert error_events[0].context == "generation"

    async def test_processor_fallback(self, vlm):
        """When apply_chat_template fails, falls back to direct processor call."""
        processor = vlm._resources.processor
        processor.apply_chat_template.side_effect = TypeError("not supported")

        input_ids = torch.tensor([[1, 2, 3]])
        processor.return_value = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

        messages = [{"role": "user", "content": "describe this"}]
        result = vlm._build_processor_inputs(messages, [])
        assert "input_ids" in result

        call_kwargs = processor.call_args.kwargs
        assert call_kwargs["text"] == "describe this"

    async def test_build_processor_inputs_passes_tools(self, vlm):
        """Tools kwarg is forwarded to apply_chat_template."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "look_up",
                    "description": "Look up info",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        messages = [{"role": "user", "content": "hi"}]
        vlm._build_processor_inputs(messages, [], tools)

        call_kwargs = vlm._resources.processor.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] is tools

    async def test_build_processor_inputs_tools_fallback(self, vlm):
        """When template fails with tools, retries without and succeeds."""
        processor = vlm._resources.processor
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if "tools" in kwargs:
                raise ValueError("tools not supported")
            ids = torch.tensor([[1, 2, 3]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        processor.apply_chat_template.side_effect = _side_effect

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = vlm._build_processor_inputs(
            [{"role": "user", "content": "hi"}], [], tools
        )
        assert "input_ids" in result
        assert call_count == 2

    async def test_tool_calls_execute_and_generate_followup(self, vlm, conversation):
        """Tool calls are executed and the VLM generates a follow-up using the same frames."""
        for _ in range(2):
            vlm._frame_buffer.append(_random_video_frame())

        tool_text = (
            '<tool_call>{"name": "identify", "arguments": {"label": "cat"}}</tool_call>'
        )
        followup_text = "That is a cat."

        # First call returns tool call text, second returns plain follow-up
        call_count = 0

        def _decode_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_text
            return followup_text

        vlm._resources.processor.decode.side_effect = _decode_side_effect

        calls_received = []

        @vlm.register_function("identify", description="Identify object")
        async def identify(label: str) -> str:
            calls_received.append(label)
            return f"Confirmed: {label}"

        frames_snapshot = list(vlm._frame_buffer)
        image_count = min(len(frames_snapshot), vlm._max_frames)
        messages = vlm._build_messages()
        image_content = [{"type": "image"} for _ in range(image_count)]
        image_content.append({"type": "text", "text": "what is this?"})
        messages.append({"role": "user", "content": image_content})

        result = await vlm.create_response(messages=messages, frames=frames_snapshot)

        assert calls_received == ["cat"]
        assert result.text == followup_text

        # Follow-up call should reuse the same frames
        last_call = vlm._resources.processor.apply_chat_template.call_args
        assert last_call.kwargs.get("images") is not None
        assert len(last_call.kwargs["images"]) == 2


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@skip_blockbuster
class TestTransformersVLMIntegration:
    async def test_simple_response(self):
        model_id = os.getenv(
            "TRANSFORMERS_TEST_VLM", "HuggingFaceTB/SmolVLM-256M-Instruct"
        )

        vlm = TransformersVLM(model=model_id, max_new_tokens=30)
        conversation = InMemoryConversation("", [])
        vlm.set_conversation(conversation)

        resources = await vlm.on_warmup()
        vlm.on_warmed_up(resources)

        # Add a frame so the VLM has something to look at
        vlm._frame_buffer.append(_random_video_frame())

        response = await vlm.simple_response(text="Describe what you see")
        assert response.text
        assert len(response.text) > 0

        vlm.unload()
