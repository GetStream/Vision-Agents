"""
TransformersVLM - Local vision-language model inference via HuggingFace Transformers.

Runs VLMs directly on your hardware for image + text understanding.

Example:
    from vision_agents.plugins.huggingface import TransformersVLM

    vlm = TransformersVLM(model="llava-hf/llava-1.5-7b-hf")

    # Smaller, faster model with quantization
    vlm = TransformersVLM(
        model="Qwen/Qwen2-VL-2B-Instruct",
        quantization="4bit",
    )
"""

import asyncio
import gc
import logging
import time
import uuid
from collections import deque
from typing import Any, Callable, Optional, cast

import av
import jinja2
import torch
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseCompletedEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.warmup import Warmable

from . import events
from ._tool_call_loop import (
    convert_tools_to_chat_completions_format,
    run_tool_call_loop,
)
from .transformers_llm import (
    DeviceType,
    QuantizationType,
    TorchDtypeType,
    extract_tool_calls_from_text,
    get_quantization_config,
    resolve_torch_dtype,
)

logger = logging.getLogger(__name__)

PLUGIN_NAME = "transformers_vlm"


class VLMResources:
    """Container for a loaded VLM model, processor, and target device."""

    def __init__(
        self,
        model: PreTrainedModel,
        processor: Any,
        device: torch.device,
    ):
        self.model = model
        self.processor = processor
        self.device = device


class TransformersVLM(VideoLLM, Warmable[VLMResources]):
    """Local VLM inference using HuggingFace Transformers.

    Unlike ``HuggingFaceVLM`` (API-based), this runs vision-language models
    directly on your hardware.

    Args:
        model: HuggingFace model ID (e.g. ``"llava-hf/llava-1.5-7b-hf"``).
        device: ``"auto"`` (recommended), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        quantization: ``"none"``, ``"4bit"``, or ``"8bit"``.
        torch_dtype: ``"auto"``, ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        trust_remote_code: Allow custom model code (default ``True`` for VLMs).
        fps: Frames per second to capture from video stream.
        frame_buffer_seconds: Seconds of frames to keep in the buffer.
        max_frames: Maximum frames to send per inference. Evenly sampled from buffer.
        max_new_tokens: Default maximum tokens to generate per response.
        max_tool_rounds: Maximum tool-call rounds per response (default 3).
    """

    def __init__(
        self,
        model: str,
        device: DeviceType = "auto",
        quantization: QuantizationType = "none",
        torch_dtype: TorchDtypeType = "auto",
        trust_remote_code: bool = True,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        max_frames: int = 4,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 3,
    ):
        super().__init__()

        self.model_id = model
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens
        self._max_tool_rounds = max_tool_rounds
        self._fps = fps
        self._max_frames = max_frames

        self._resources: Optional[VLMResources] = None

        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )

        self.events.register_events_from_module(events)

    async def on_warmup(self) -> VLMResources:
        logger.info(f"Loading VLM: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"VLM loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: VLMResources) -> None:
        self._resources = resource

    def _load_model_sync(self) -> VLMResources:
        torch_dtype = resolve_torch_dtype(self._torch_dtype_config)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if self._device_config == "auto":
            load_kwargs["device_map"] = "auto"
        elif self._device_config == "cuda":
            load_kwargs["device_map"] = {"": "cuda"}

        quant_config = get_quantization_config(self._quantization)
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **load_kwargs
        )

        if self._device_config == "mps":
            cast(torch.nn.Module, model).to(torch.device("mps"))

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self._trust_remote_code
        )

        device = next(model.parameters()).device
        return VLMResources(model=model, processor=processor, device=device)

    async def watch_video_track(
        self,
        track: MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping the previous one")
            await self._video_forwarder.stop()
            self._video_forwarder = None

        logger.info(f'Subscribing plugin "{PLUGIN_NAME}" to VideoForwarder')

        if shared_forwarder:
            self._video_forwarder = shared_forwarder
        else:
            self._video_forwarder = VideoForwarder(
                cast(VideoStreamTrack, track),
                max_buffer=10,
                fps=self._fps,
                name=f"{PLUGIN_NAME}_forwarder",
            )
            self._video_forwarder.start()

        self._video_forwarder.add_frame_handler(
            self._frame_buffer.append, fps=self._fps
        )

    async def stop_watching_video_track(self) -> None:
        if self._video_forwarder is not None:
            await self._video_forwarder.remove_frame_handler(self._frame_buffer.append)
            self._video_forwarder = None
            logger.info(f"Stopped video forwarding to {PLUGIN_NAME}")

    async def simple_response(
        self,
        text: str,
        participant: Optional[Any] = None,
    ) -> LLMResponseEvent:
        if self._conversation is None:
            logger.warning(
                "Conversation not initialized. Call set_conversation() first."
            )
            return LLMResponseEvent(original=None, text="")

        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        frames_snapshot = list(self._frame_buffer)
        image_count = min(len(frames_snapshot), self._max_frames)

        messages = self._build_messages()
        image_content: list[dict[str, Any]] = [
            {"type": "image"} for _ in range(image_count)
        ]
        image_content.append({"type": "text", "text": text or "Describe what you see."})
        messages.append({"role": "user", "content": image_content})

        inference_id = str(uuid.uuid4())

        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model_id,
                frames_count=len(frames_snapshot),
            )
        )

        request_start = time.perf_counter()
        response = await self.create_response(messages=messages, frames=frames_snapshot)
        latency_ms = (time.perf_counter() - request_start) * 1000

        if response.exception is not None:
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=response.exception,
                    context="generation",
                )
            )
        else:
            self.events.send(
                VLMInferenceCompletedEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    model=self.model_id,
                    text=response.text,
                    latency_ms=latency_ms,
                    frames_processed=len(frames_snapshot),
                )
            )

        return response

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        frames: Optional[list[av.VideoFrame]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        """Generate a response from messages and optional video frames.

        Args:
            messages: Chat messages. If ``None``, builds from conversation history.
            frames: Video frames to include. If ``None``, uses the current buffer.
            max_new_tokens: Override the default max token count.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (vs greedy).
        """
        is_tool_followup = kwargs.pop("_tool_followup", False)

        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if messages is None:
            messages = self._build_messages()
        if frames is None:
            frames = list(self._frame_buffer)

        tools_param: Optional[list[dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_chat_completions_format(tools_spec)

        try:
            inputs = await asyncio.to_thread(
                self._build_processor_inputs, messages, frames, tools_param
            )
        except (jinja2.TemplateError, TypeError, ValueError, RuntimeError) as e:
            logger.exception("Failed to build VLM inputs")
            return LLMResponseEvent(original=None, text="", exception=e)

        device = self._resources.device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        max_tokens = max_new_tokens or self._max_new_tokens
        model = self._resources.model
        processor = self._resources.processor
        pad_token_id = processor.tokenizer.pad_token_id

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=self.model_id,
                streaming=False,
            )
        )

        request_start = time.perf_counter()

        def _do_generate() -> Any:
            gen_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "temperature": temperature if do_sample else 1.0,
            }
            if pad_token_id is not None:
                gen_kwargs["pad_token_id"] = pad_token_id
            with torch.no_grad():
                return cast(Callable[..., torch.Tensor], model.generate)(**gen_kwargs)

        try:
            outputs = await asyncio.to_thread(_do_generate)
        except RuntimeError as e:
            logger.exception("VLM generation failed")
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="", exception=e)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        if tools_param and output_text:
            tool_calls = extract_tool_calls_from_text(output_text)
            if tool_calls:
                if is_tool_followup:
                    # Return to run_tool_call_loop — it will handle these
                    # tool calls in the next round without nesting loops.
                    return LLMResponseEvent(original=outputs, text=output_text)
                return await self._handle_tool_calls(
                    tool_calls, messages, frames, kwargs
                )

        latency_ms = (time.perf_counter() - request_start) * 1000
        response_id = str(uuid.uuid4())

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=outputs,
                text=output_text,
                item_id=response_id,
                latency_ms=latency_ms,
                model=self.model_id,
            )
        )

        return LLMResponseEvent(original=outputs, text=output_text)

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        if self._conversation:
            for msg in self._conversation.messages:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def _build_processor_inputs(
        self,
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Build processor inputs from messages, video frames, and optional tools.

        Samples frames evenly to stay within ``max_frames``, converts them to
        PIL images, then applies the processor's chat template.
        """
        assert self._resources is not None
        processor = self._resources.processor

        all_frames = list(frames)
        if len(all_frames) > self._max_frames:
            step = len(all_frames) / self._max_frames
            all_frames = [all_frames[int(i * step)] for i in range(self._max_frames)]

        images = [frame.to_image() for frame in all_frames]

        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if tools:
            template_kwargs["tools"] = tools

        try:
            result = processor.apply_chat_template(
                messages,
                images=images if images else None,
                **template_kwargs,
            )
            if isinstance(result, str):
                return processor(
                    text=result,
                    images=images if images else None,
                    return_tensors="pt",
                    padding=True,
                )
            return result
        except (jinja2.TemplateError, TypeError, ValueError) as e:
            if tools:
                logger.warning(
                    f"apply_chat_template failed with tools, retrying without: {e}"
                )
                template_kwargs.pop("tools", None)
                result = processor.apply_chat_template(
                    messages,
                    images=images if images else None,
                    **template_kwargs,
                )
                if isinstance(result, str):
                    return processor(
                        text=result,
                        images=images if images else None,
                        return_tensors="pt",
                        padding=True,
                    )
                return result

            logger.warning(f"processor.apply_chat_template failed, using fallback: {e}")
            prompt = "Describe what you see."
            if messages:
                last_content = messages[-1].get("content", prompt)
                if isinstance(last_content, str):
                    prompt = last_content
                elif isinstance(last_content, list):
                    for item in last_content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt = item.get("text", prompt)
                            break
            return processor(
                text=prompt,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        return convert_tools_to_chat_completions_format(tools)

    async def _handle_tool_calls(
        self,
        tool_calls: list[NormalizedToolCallItem],
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        kwargs: dict[str, Any],
    ) -> LLMResponseEvent:
        """Execute tool calls and generate follow-up responses with the same frames."""

        async def _generate_followup(
            current_messages: list[dict[str, Any]],
        ) -> tuple[LLMResponseEvent, list[NormalizedToolCallItem]]:
            result = await self.create_response(
                messages=current_messages, frames=frames, _tool_followup=True, **kwargs
            )
            next_calls = extract_tool_calls_from_text(result.text)
            return result, next_calls

        return await run_tool_call_loop(
            self,
            tool_calls,
            messages,
            _generate_followup,
            max_rounds=self._max_tool_rounds,
        )

    def unload(self) -> None:
        logger.info(f"Unloading VLM: {self.model_id}")
        if self._resources is not None:
            del self._resources.model
            del self._resources.processor
            self._resources = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

    @property
    def is_loaded(self) -> bool:
        return self._resources is not None

    @property
    def device(self) -> Optional[torch.device]:
        if self._resources:
            return self._resources.device
        return None
