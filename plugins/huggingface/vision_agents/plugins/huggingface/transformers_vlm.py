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

import gc
import logging
from typing import Any, Callable, Optional, cast

import av
import jinja2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel
from ._local_vlm import LocalVLM
from .transformers_llm import (
    DeviceType,
    QuantizationType,
    TorchDtypeType,
    get_quantization_config,
    resolve_device,
    resolve_torch_dtype,
)

logger = logging.getLogger(__name__)


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


class TransformersVLM(LocalVLM[VLMResources]):
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

    _plugin_name = "transformers_vlm"

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
        do_sample: bool = True,
    ):
        super().__init__(
            model,
            fps,
            frame_buffer_seconds,
            max_frames,
            max_new_tokens,
            max_tool_rounds,
        )
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._do_sample = do_sample

    def _load_model_sync(self) -> VLMResources:
        torch_dtype = resolve_torch_dtype(self._torch_dtype_config)
        device = resolve_device(self._device_config)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if device == "cuda":
            load_kwargs["device_map"] = {"": "cuda"}

        quant_config = get_quantization_config(self._quantization)
        if quant_config:
            load_kwargs["quantization_config"] = quant_config

        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, **load_kwargs
        )

        if device == "mps":
            cast(torch.nn.Module, model).to(torch.device("mps"))

        model.eval()

        processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self._trust_remote_code
        )

        device = next(model.parameters()).device
        return VLMResources(model=model, processor=processor, device=device)

    def _generate_with_frames(
        self,
        messages: list[dict[str, Any]],
        frames: list[av.VideoFrame],
        tools_param: Optional[list[dict[str, Any]]],
        max_tokens: int,
        temperature: float,
    ) -> str:
        assert self._resources is not None
        model = self._resources.model
        processor = self._resources.processor
        device = self._resources.device

        all_frames = list(frames)
        if len(all_frames) > self._max_frames:
            step = len(all_frames) / self._max_frames
            all_frames = [all_frames[int(i * step)] for i in range(self._max_frames)]

        images = [frame.to_image() for frame in all_frames]

        inputs = self._build_processor_inputs(processor, messages, images, tools_param)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        pad_token_id = processor.tokenizer.pad_token_id
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": self._do_sample,
            "temperature": temperature if self._do_sample else 1.0,
        }
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id

        with torch.no_grad():
            outputs = cast(Callable[..., torch.Tensor], model.generate)(**gen_kwargs)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        return processor.decode(generated_ids, skip_special_tokens=True)

    @staticmethod
    def _build_processor_inputs(
        processor: Any,
        messages: list[dict[str, Any]],
        images: list[Any],
        tools: Optional[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Build processor inputs from messages, images, and optional tools."""
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
                    "apply_chat_template failed with tools, retrying without: %s", e
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

            logger.warning(
                "processor.apply_chat_template failed, using fallback: %s", e
            )
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

    def unload(self) -> None:
        logger.info("Unloading VLM: %s", self.model_id)
        if self._resources is not None:
            del self._resources.model
            del self._resources.processor
            self._resources = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

    @property
    def device(self) -> Optional[torch.device]:
        if self._resources:
            return self._resources.device
        return None
