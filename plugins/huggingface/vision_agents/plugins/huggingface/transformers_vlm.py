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
import json
import logging
import time
import uuid
from collections import deque
from typing import Any, AsyncIterator, Callable, Optional, cast

import av
import jinja2
import torch
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    StoppingCriteriaList,
)
from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal, VideoLLM
from vision_agents.core.llm.llm_types import ToolSchema
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.warmup import Warmable

from . import events
from ._tool_call_loop import convert_tools_to_chat_completions_format
from .transformers_llm import (
    DeviceType,
    QuantizationType,
    TorchDtypeType,
    _CancelStoppingCriteria,
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
        self._stopping_criteria = _CancelStoppingCriteria()

        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )

        self.events.register_events_from_module(events)

    async def interrupt(self) -> None:
        """Stop any in-flight ``model.generate`` call within ≤1 token."""
        await super().interrupt()
        self._stopping_criteria.cancel()

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
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        if self._conversation is None:
            logger.warning(
                "Conversation not initialized. Call set_conversation() first."
            )
            yield LLMResponseFinal(original=None, text="")
            return

        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            yield LLMResponseFinal(original=None, text="")
            return

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

        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=str(uuid.uuid4()),
                model=self.model_id,
                frames_count=len(frames_snapshot),
            )
        )

        async for item in self.create_response(messages=messages, frames=frames_snapshot):
            yield item

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        frames: Optional[list[av.VideoFrame]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Generate a response from messages and optional video frames.

        Args:
            messages: Chat messages. If ``None``, builds from conversation history.
            frames: Video frames to include. If ``None``, uses the current buffer.
            max_new_tokens: Override the default max token count.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (vs greedy).
        """
        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            yield LLMResponseFinal(original=None, text="")
            return

        if messages is None:
            messages = self._build_messages()
        if frames is None:
            frames = list(self._frame_buffer)

        # Reset cancellation for the new turn. ``interrupt()`` flips
        # the criterion; ``model.generate`` checks it between tokens.
        self._stopping_criteria.reset()
        stopping_criteria = StoppingCriteriaList([self._stopping_criteria])

        tools_param: Optional[list[dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_chat_completions_format(tools_spec)

        device = self._resources.device
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

        current_messages = list(messages)
        seen: set[tuple[str | None, str, str]] = set()

        for round_num in range(self._max_tool_rounds):
            try:
                inputs = await asyncio.to_thread(
                    self._build_processor_inputs,
                    current_messages,
                    frames,
                    tools_param,
                )
            except (jinja2.TemplateError, TypeError, ValueError, RuntimeError):
                logger.exception("Failed to build VLM inputs")
                yield LLMResponseFinal(original=None, text="")
                return

            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            request_start = time.perf_counter()

            def _do_generate() -> Any:
                gen_kwargs: dict[str, Any] = {
                    **inputs,
                    "max_new_tokens": max_tokens,
                    "stopping_criteria": stopping_criteria,
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                }
                if pad_token_id is not None:
                    gen_kwargs["pad_token_id"] = pad_token_id
                with torch.no_grad():
                    return cast(Callable[..., torch.Tensor], model.generate)(
                        **gen_kwargs
                    )

            try:
                outputs = await asyncio.to_thread(_do_generate)
            except RuntimeError:
                logger.exception("VLM generation failed")
                yield LLMResponseFinal(original=None, text="")
                return

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            output_text = processor.decode(generated_ids, skip_special_tokens=True)
            latency_ms = (time.perf_counter() - request_start) * 1000

            tool_calls = (
                extract_tool_calls_from_text(output_text)
                if tools_param and output_text
                else []
            )
            if not tool_calls:
                response_id = str(uuid.uuid4())
                if output_text:
                    yield LLMResponseDelta(
                        content_index=None,
                        item_id=response_id,
                        output_index=0,
                        sequence_number=0,
                        delta=output_text,
                        is_first_chunk=True,
                        time_to_first_token_ms=None,
                    )
                yield LLMResponseFinal(
                    original=outputs,
                    text=output_text,
                    item_id=response_id,
                    latency_ms=latency_ms,
                    model=self.model_id,
                )
                return

            logger.info(
                "Tool call round %d: executing %d call(s) — %s",
                round_num + 1,
                len(tool_calls),
                ", ".join(tc.get("name", "?") for tc in tool_calls),
            )
            triples, seen = await self._dedup_and_execute(
                tool_calls, max_concurrency=8, timeout_s=30, seen=seen
            )
            if not triples:
                yield LLMResponseFinal(original=None, text="")
                return

            assistant_tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for call_index, (tc, res, err) in enumerate(triples):
                cid = tc.get("id") or f"tool_call_{round_num}_{call_index}"
                name = tc["name"]
                args = tc.get("arguments_json", {})
                if err is not None:
                    logger.warning("  [tool] %s(%s) failed: %s", name, args, err)
                else:
                    logger.info("  [tool] %s(%s) → %s", name, args, res)
                assistant_tool_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(tc.get("arguments_json", {})),
                        },
                    }
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": cid,
                        "content": self._sanitize_tool_output(
                            err if err is not None else res
                        ),
                    }
                )
            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

        # Max rounds exhausted without a final answer.
        yield LLMResponseFinal(original=None, text="")

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
