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

from __future__ import annotations

import asyncio
import gc
import logging
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

import av
import torch
from aiortc.mediastreams import MediaStreamTrack, VideoStreamTrack

from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseCompletedEvent,
    VLMErrorEvent,
    VLMInferenceCompletedEvent,
    VLMInferenceStartEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent, VideoLLM
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.warmup import Warmable

from . import events
from .transformers_llm import (
    DeviceType,
    QuantizationType,
    get_quantization_config,
    resolve_torch_dtype,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from vision_agents.core.processors import Processor

logger = logging.getLogger(__name__)

PLUGIN_NAME = "transformers_vlm"


# ---------------------------------------------------------------------------
# Resource container
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TransformersVLM
# ---------------------------------------------------------------------------


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
        max_new_tokens: Default maximum tokens to generate per response.
    """

    def __init__(
        self,
        model: str,
        device: DeviceType = "auto",
        quantization: QuantizationType = "none",
        torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto",
        trust_remote_code: bool = True,
        fps: int = 1,
        frame_buffer_seconds: int = 10,
        max_new_tokens: int = 512,
    ):
        VideoLLM.__init__(self)

        self.model_id = model
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens
        self._fps = fps

        self._resources: Optional[VLMResources] = None

        # Video frame handling (mirrors HuggingFaceVLM)
        self._video_forwarder: Optional[VideoForwarder] = None
        self._frame_buffer: deque[av.VideoFrame] = deque(
            maxlen=fps * frame_buffer_seconds
        )
        self._frame_width = 800
        self._frame_height = 600

        self.events.register_events_from_module(events)

    # ------------------------------------------------------------------
    # Warmable interface
    # ------------------------------------------------------------------

    async def on_warmup(self) -> VLMResources:
        logger.info(f"Loading VLM: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"VLM loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: VLMResources) -> None:
        self._resources = resource

    def _load_model_sync(self) -> VLMResources:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch_dtype = resolve_torch_dtype(self._torch_dtype_config)

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "dtype": torch_dtype,
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
            model = model.to("mps")

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self._trust_remote_code
        )

        device = next(model.parameters()).device
        return VLMResources(model=model, processor=processor, device=device)

    # ------------------------------------------------------------------
    # VideoLLM interface
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # LLM interface
    # ------------------------------------------------------------------

    async def simple_response(
        self,
        text: str,
        processors: Optional[list[Processor]] = None,
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

        frames_count = len(self._frame_buffer)
        inference_id = str(uuid.uuid4())

        self.events.send(
            VLMInferenceStartEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model_id,
                frames_count=frames_count,
            )
        )
        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=self.model_id,
                streaming=False,
            )
        )

        request_start = time.perf_counter()

        try:
            inputs = self._build_vlm_inputs(text)
        except Exception as e:
            logger.error(f"Failed to build VLM inputs: {e}")
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=e,
                    context="input_processing",
                )
            )
            return LLMResponseEvent(original=None, text="")

        # Move tensors to device
        device = self._resources.device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        processor = self._resources.processor
        model = self._resources.model

        pad_token_id = processor.tokenizer.pad_token_id

        def _do_generate() -> Any:
            gen_kwargs: Dict[str, Any] = {
                **inputs,
                "max_new_tokens": self._max_new_tokens,
            }
            if pad_token_id is not None:
                gen_kwargs["pad_token_id"] = pad_token_id
            with torch.no_grad():
                return model.generate(**gen_kwargs)

        try:
            outputs = await asyncio.to_thread(_do_generate)
        except Exception as e:
            logger.error(f"VLM generation failed: {e}")
            self.events.send(
                VLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    inference_id=inference_id,
                    error=e,
                    context="generation",
                )
            )
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        # Decode only newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)

        latency_ms = (time.perf_counter() - request_start) * 1000

        self.events.send(
            VLMInferenceCompletedEvent(
                plugin_name=PLUGIN_NAME,
                inference_id=inference_id,
                model=self.model_id,
                text=output_text,
                latency_ms=latency_ms,
                frames_processed=frames_count,
            )
        )
        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=outputs,
                text=output_text,
                item_id=inference_id,
                latency_ms=latency_ms,
                model=self.model_id,
            )
        )

        return LLMResponseEvent(original=outputs, text=output_text)

    # ------------------------------------------------------------------
    # Input building
    # ------------------------------------------------------------------

    def _build_vlm_inputs(self, text: str) -> Dict[str, Any]:
        """Build processor inputs from text and buffered video frames.

        Converts ``av.VideoFrame`` objects to PIL Images and passes them
        to the processor alongside a structured message list.
        """
        processor = self._resources.processor  # type: ignore[union-attr]

        # Convert buffered frames to PIL images
        images = [frame.to_image() for frame in self._frame_buffer]

        # Build chat messages
        messages: List[Dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        if self._conversation:
            for msg in self._conversation.messages:
                messages.append({"role": msg.role, "content": msg.content})

        # User message with image placeholders + text
        user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": text or "Describe what you see."})
        messages.append({"role": "user", "content": user_content})

        # Apply chat template to get formatted prompt, then tokenize.
        # Some processors return tokenized tensors directly from
        # apply_chat_template (return_dict=True), others return a string
        # that needs a separate processor() call to tokenize.
        try:
            result = processor.apply_chat_template(
                messages,
                images=images if images else None,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if isinstance(result, str):
                return processor(
                    text=result,
                    images=images if images else None,
                    return_tensors="pt",
                    padding=True,
                )
            return result
        except Exception as e:
            logger.warning(
                f"processor.apply_chat_template failed, using fallback: {e}"
            )
            prompt = text or "Describe what you see."
            return processor(
                text=prompt,
                images=images if images else None,
                return_tensors="pt",
                padding=True,
            )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

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
