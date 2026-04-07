"""
TransformersLLM - Local text LLM inference via HuggingFace Transformers.

Runs models directly on your hardware (GPU/CPU/MPS) instead of calling APIs.

Example:
    from vision_agents.plugins.huggingface import TransformersLLM

    llm = TransformersLLM(model="meta-llama/Llama-3.2-3B-Instruct")

    # With 4-bit quantization (~4x memory reduction)
    llm = TransformersLLM(
        model="meta-llama/Llama-3.2-3B-Instruct",
        quantization="4bit",
    )
"""

import asyncio
import gc
import logging
import time
import uuid
from threading import Thread
from typing import Any, Callable, Literal, Optional, cast

import jinja2
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent

from . import events
from ._local_inference import LocalTextLLM

# Re-exported for backward compatibility (tests import from this module).
from ._local_inference import (
    extract_tool_calls_from_text as extract_tool_calls_from_text,
)

logger = logging.getLogger(__name__)

DeviceType = Literal["auto", "cuda", "mps", "cpu"]
QuantizationType = Literal["none", "4bit", "8bit"]
TorchDtypeType = Literal["auto", "float16", "bfloat16", "float32"]


def resolve_device(config: DeviceType) -> DeviceType:
    """Resolve ``"auto"`` to a concrete device based on available hardware.

    ``device_map="auto"`` in HuggingFace only handles CPU/CUDA splits and does
    not place models on MPS.  This helper picks the best concrete device so
    callers can use the MPS path explicitly when running on Apple Silicon.
    """
    if config != "auto":
        return config
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(config: TorchDtypeType) -> torch.dtype:
    """Map a string config to a concrete ``torch.dtype``.

    When *config* is ``"auto"`` the best dtype is chosen based on available
    hardware: ``bfloat16`` on CUDA with bf16 support, ``float16`` on CUDA/MPS,
    and ``float32`` on CPU.
    """
    if config == "float16":
        return torch.float16
    if config == "bfloat16":
        return torch.bfloat16
    if config == "float32":
        return torch.float32
    # "auto"
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def get_quantization_config(quantization: QuantizationType) -> Optional[Any]:
    """Build a ``BitsAndBytesConfig`` for 4-bit / 8-bit quantization.

    Returns ``None`` when *quantization* is ``"none"``.
    """
    if quantization == "none":
        return None

    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


class ModelResources:
    """Container for a loaded model, tokenizer, and target device."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


class TransformersLLM(LocalTextLLM[ModelResources]):
    """Local LLM inference using HuggingFace Transformers.

    Unlike ``HuggingFaceLLM`` (API-based), this runs models directly on your
    hardware.

    Args:
        model: HuggingFace model ID (e.g. ``"meta-llama/Llama-3.2-3B-Instruct"``).
        device: ``"auto"`` (recommended), ``"cuda"``, ``"mps"``, or ``"cpu"``.
        quantization: ``"none"``, ``"4bit"``, or ``"8bit"``.
        torch_dtype: ``"auto"``, ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        trust_remote_code: Allow custom model code (needed for Qwen, Phi, etc.).
        max_new_tokens: Default maximum tokens to generate per response.
        max_tool_rounds: Maximum tool-call rounds per response (default 3).
    """

    _plugin_name = "transformers_llm"

    def __init__(
        self,
        model: str,
        device: DeviceType = "auto",
        quantization: QuantizationType = "none",
        torch_dtype: TorchDtypeType = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 3,
    ):
        super().__init__(model, max_new_tokens, max_tool_rounds)
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code

    def _load_model_sync(self) -> ModelResources:
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

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        if device == "mps":
            cast(torch.nn.Module, model).to(torch.device("mps"))

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self._trust_remote_code,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_device = next(model.parameters()).device
        return ModelResources(model=model, tokenizer=tokenizer, device=torch_device)

    def _apply_template(
        self,
        messages: list[dict[str, Any]],
        tools_param: Optional[list[dict[str, Any]]],
    ) -> tuple[Any, bool] | None:
        assert self._resources is not None
        tokenizer = self._resources.tokenizer
        device = self._resources.device

        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if tools_param:
            template_kwargs["tools"] = tools_param

        try:
            inputs = cast(
                dict[str, Any],
                tokenizer.apply_chat_template(messages, **template_kwargs),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            return inputs, tools_param is not None
        except (jinja2.TemplateError, TypeError, ValueError) as e:
            if tools_param:
                logger.warning(
                    "apply_chat_template failed with tools, retrying without: %s", e
                )
                template_kwargs.pop("tools", None)
                inputs = cast(
                    dict[str, Any],
                    tokenizer.apply_chat_template(messages, **template_kwargs),
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                return inputs, False
            logger.exception("Failed to apply chat template")
            return None

    async def _generate_streaming(
        self, prepared_input: Any, max_tokens: int, temperature: float
    ) -> LLMResponseEvent:
        assert self._resources is not None
        model = self._resources.model
        tokenizer = self._resources.tokenizer

        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        class _AsyncBridgeStreamer(TextStreamer):
            """Bridges token text to an ``asyncio.Queue``."""

            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                loop.call_soon_threadsafe(async_queue.put_nowait, text)
                if stream_end:
                    loop.call_soon_threadsafe(async_queue.put_nowait, None)

        streamer = _AsyncBridgeStreamer(
            cast(PreTrainedTokenizerBase, tokenizer),
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generate_kwargs = {
            **prepared_input,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": True,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
        }

        request_start = time.perf_counter()
        first_token_time: Optional[float] = None
        text_chunks: list[str] = []
        chunk_index = 0
        response_id = str(uuid.uuid4())
        generation_error: Optional[Exception] = None

        def run_generation() -> None:
            nonlocal generation_error
            try:
                with torch.no_grad():
                    cast(Callable[..., torch.Tensor], model.generate)(**generate_kwargs)
            except RuntimeError as e:
                generation_error = e
                logger.exception("Generation failed")
            finally:
                loop.call_soon_threadsafe(async_queue.put_nowait, None)

        thread = Thread(target=run_generation, daemon=True)
        thread.start()

        while True:
            item = await async_queue.get()
            if item is None:
                break

            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - request_start) * 1000
            else:
                ttft_ms = None

            text_chunks.append(item)

            self.events.send(
                LLMResponseChunkEvent(
                    plugin_name=self._plugin_name,
                    content_index=None,
                    item_id=response_id,
                    output_index=0,
                    sequence_number=chunk_index,
                    delta=item,
                    is_first_chunk=(chunk_index == 0),
                    time_to_first_token_ms=ttft_ms,
                )
            )
            chunk_index += 1

        thread.join(timeout=5.0)

        if generation_error:
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=self._plugin_name,
                    error_message=str(generation_error),
                    event_data=generation_error,
                )
            )
            return LLMResponseEvent(original=None, text="")

        total_text = "".join(text_chunks)
        latency_ms = (time.perf_counter() - request_start) * 1000
        ttft_final = (
            (first_token_time - request_start) * 1000 if first_token_time else None
        )

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=self._plugin_name,
                original=None,
                text=total_text,
                item_id=response_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=ttft_final,
                model=self.model_id,
            )
        )

        return LLMResponseEvent(original=None, text=total_text)

    async def _generate_non_streaming(
        self,
        prepared_input: Any,
        max_tokens: int,
        temperature: float,
        emit_events: bool,
    ) -> LLMResponseEvent:
        assert self._resources is not None
        model = self._resources.model
        tokenizer = self._resources.tokenizer

        request_start = time.perf_counter()
        response_id = str(uuid.uuid4())

        generate_kwargs = {
            **prepared_input,
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": temperature,
            "pad_token_id": tokenizer.pad_token_id,
        }

        def _do_generate() -> Any:
            with torch.no_grad():
                return cast(Callable[..., torch.Tensor], model.generate)(
                    **generate_kwargs
                )

        try:
            outputs = await asyncio.to_thread(_do_generate)
        except RuntimeError as e:
            logger.exception("Generation failed")
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=self._plugin_name,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        input_length = prepared_input["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        text = str(tokenizer.decode(generated_ids, skip_special_tokens=True))

        if emit_events:
            latency_ms = (time.perf_counter() - request_start) * 1000
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name=self._plugin_name,
                    original=outputs,
                    text=text,
                    item_id=response_id,
                    latency_ms=latency_ms,
                    model=self.model_id,
                )
            )

        return LLMResponseEvent(original=outputs, text=text)

    def unload(self) -> None:
        logger.info("Unloading model: %s", self.model_id)
        if self._resources is not None:
            del self._resources.model
            del self._resources.tokenizer
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
