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
import json
import logging
import re
import time
import uuid
from threading import Thread
from typing import Any, AsyncIterator, Callable, Literal, Optional, cast

import jinja2
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)
from vision_agents.core.llm.llm import LLM, LLMResponseDelta, LLMResponseFinal
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.warmup import Warmable

from . import events
from ._tool_call_loop import convert_tools_to_chat_completions_format

logger = logging.getLogger(__name__)

PLUGIN_NAME = "transformers_llm"

DeviceType = Literal["auto", "cuda", "mps", "cpu"]
QuantizationType = Literal["none", "4bit", "8bit"]
TorchDtypeType = Literal["auto", "float16", "bfloat16", "float32"]


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


convert_tools_to_transformers_format = convert_tools_to_chat_completions_format


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract all top-level JSON objects from *text* using ``raw_decode``.

    Handles arbitrarily nested braces, unlike a regex approach.
    """
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        idx = text.find("{", idx)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return objects


def extract_tool_calls_from_text(text: str) -> list[NormalizedToolCallItem]:
    """Parse tool calls from raw model output text.

    Supports:
    - Hermes format: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
    - Generic JSON: ``{"name": ..., "arguments": ...}``
    """
    tool_calls: list[NormalizedToolCallItem] = []

    hermes_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    for match in re.finditer(hermes_pattern, text, re.DOTALL):
        for obj in _extract_json_objects(match.group(1)):
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": obj.get("id", str(uuid.uuid4())),
                    "name": obj.get("name", ""),
                    "arguments_json": obj.get("arguments", {}),
                }
            )

    if tool_calls:
        return tool_calls

    for obj in _extract_json_objects(text):
        if "name" in obj and "arguments" in obj:
            tool_calls.append(
                {
                    "type": "tool_call",
                    "id": str(uuid.uuid4()),
                    "name": obj["name"],
                    "arguments_json": obj["arguments"],
                }
            )

    return tool_calls


class _CancelStoppingCriteria(StoppingCriteria):
    """``model.generate`` consults this between tokens; calling ``cancel()``
    ends generation within ≤1 token. ``reset()`` clears the flag for the next
    generation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def reset(self) -> None:
        self._cancelled = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        return self._cancelled


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


class TransformersLLM(LLM, Warmable[ModelResources]):
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

    provider_name = PLUGIN_NAME

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
        super().__init__()

        self.model_id = model
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens
        self._max_tool_rounds = max_tool_rounds

        self._resources: Optional[ModelResources] = None
        self._stopping_criteria = _CancelStoppingCriteria()

        self.events.register_events_from_module(events)

    async def interrupt(self) -> None:
        """Stop any in-flight ``model.generate`` call within ≤1 token."""
        await super().interrupt()
        self._stopping_criteria.cancel()

    async def on_warmup(self) -> ModelResources:
        logger.info(f"Loading model: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"Model loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: ModelResources) -> None:
        self._resources = resource

    def _load_model_sync(self) -> ModelResources:
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

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        if self._device_config == "mps":
            cast(torch.nn.Module, model).to(torch.device("mps"))

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self._trust_remote_code,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        return ModelResources(model=model, tokenizer=tokenizer, device=device)

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

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = self._build_messages()
        async for item in self.create_response(messages=messages, stream=True):
            yield item

    async def create_response(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        *,
        stream: bool = True,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            yield LLMResponseFinal(original=None, text="")
            return

        if messages is None:
            messages = self._build_messages()

        # Reset cancellation for the new turn. ``interrupt()`` flips
        # the criterion; ``model.generate`` checks it between tokens.
        self._stopping_criteria.reset()
        stopping_criteria = StoppingCriteriaList([self._stopping_criteria])

        model = self._resources.model
        tokenizer = self._resources.tokenizer
        device = self._resources.device

        tools_param: Optional[list[dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = convert_tools_to_transformers_format(tools_spec)

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
        except (jinja2.TemplateError, TypeError, ValueError) as e:
            if tools_param:
                logger.warning(
                    f"apply_chat_template failed with tools, retrying without: {e}"
                )
                template_kwargs.pop("tools", None)
                inputs = cast(
                    dict[str, Any],
                    tokenizer.apply_chat_template(messages, **template_kwargs),
                )
                tools_param = None
            else:
                logger.exception("Failed to apply chat template")
                yield LLMResponseFinal(original=None, text="")
                return

        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_tokens = max_new_tokens or self._max_new_tokens

        if tools_param is None:
            # No tools — stream or generate normally and yield through.
            gen = (
                self._generate_streaming(
                    model,
                    tokenizer,
                    inputs,
                    max_tokens,
                    temperature,
                    do_sample,
                    stopping_criteria,
                )
                if stream
                else self._generate_non_streaming(
                    model,
                    tokenizer,
                    inputs,
                    max_tokens,
                    temperature,
                    do_sample,
                    stopping_criteria,
                )
            )
            async for item in gen:
                yield item
            return

        # Tools registered — force non-streaming and run an inline tool-call
        # loop. Raw model output may contain tool-call markup (e.g.
        # <tool_call>…</tool_call>) that must not be yielded as user-visible
        # deltas; only the final natural-language answer is yielded.

        current_messages = list(messages)
        current_inputs = inputs
        seen: set[tuple[str | None, str, str]] = set()

        for round_num in range(self._max_tool_rounds + 1):
            text = ""
            original: Any = None
            async for item in self._generate_non_streaming(
                model,
                tokenizer,
                current_inputs,
                max_tokens,
                temperature,
                do_sample,
                stopping_criteria,
            ):
                if isinstance(item, LLMResponseFinal):
                    text = item.text
                    original = item.original

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Raw model output (tools registered): %s", text)

            if not text:
                yield LLMResponseFinal(original=None, text="")
                return

            tool_calls = extract_tool_calls_from_text(text)
            if not tool_calls:
                response_id = str(uuid.uuid4())
                yield LLMResponseDelta(
                    content_index=None,
                    item_id=response_id,
                    output_index=0,
                    sequence_number=0,
                    delta=text,
                    is_first_chunk=True,
                    time_to_first_token_ms=None,
                )
                yield LLMResponseFinal(
                    original=original,
                    text=text,
                    item_id=response_id,
                    model=self.model_id,
                )
                return

            if round_num >= self._max_tool_rounds:
                break

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

            round_template_kwargs: dict[str, Any] = {
                "add_generation_prompt": True,
                "return_dict": True,
                "return_tensors": "pt",
                "tools": tools_param,
            }
            next_inputs = cast(
                dict[str, Any],
                tokenizer.apply_chat_template(
                    current_messages, **round_template_kwargs
                ),
            )
            current_inputs = {k: v.to(device) for k, v in next_inputs.items()}

        # Max rounds exhausted without a final answer.
        yield LLMResponseFinal(original=None, text="")

    async def _generate_streaming(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stopping_criteria: StoppingCriteriaList,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        class _AsyncBridgeStreamer(TextStreamer):
            """Bridges token text to an ``asyncio.Queue`` without blocking the event loop."""

            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                if text:
                    loop.call_soon_threadsafe(async_queue.put_nowait, text)
                if stream_end:
                    loop.call_soon_threadsafe(async_queue.put_nowait, None)

        streamer = _AsyncBridgeStreamer(
            cast(AutoTokenizer, tokenizer),
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
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
                self.on_llm_error(error=e)
            finally:
                # Flush any text still buffered inside the TextStreamer
                # (everything after the last word boundary). model.generate
                # does not call streamer.end() itself, so without this we
                # silently drop the tail of the response.
                streamer.end()
                loop.call_soon_threadsafe(async_queue.put_nowait, None)

        thread = Thread(target=run_generation, daemon=True)
        thread.start()

        while True:
            item = await async_queue.get()
            if item is None:
                break

            is_first = first_token_time is None
            ttft_ms: Optional[float] = None
            if is_first:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - request_start) * 1000

            text_chunks.append(item)
            yield LLMResponseDelta(
                content_index=None,
                item_id=response_id,
                output_index=0,
                sequence_number=chunk_index,
                delta=item,
                is_first_chunk=is_first,
                time_to_first_token_ms=ttft_ms,
            )
            chunk_index += 1

        thread.join(timeout=5.0)

        if generation_error:
            yield LLMResponseFinal(original=None, text="")
            return

        total_text = "".join(text_chunks)
        latency_ms = (time.perf_counter() - request_start) * 1000
        ttft_final = (
            (first_token_time - request_start) * 1000 if first_token_time else None
        )

        yield LLMResponseFinal(
            original=None,
            text=total_text,
            item_id=response_id,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_final,
            model=self.model_id,
        )

    async def _generate_non_streaming(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs: dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        stopping_criteria: StoppingCriteriaList,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        request_start = time.perf_counter()
        response_id = str(uuid.uuid4())

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "stopping_criteria": stopping_criteria,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "pad_token_id": tokenizer.pad_token_id,
        }

        def _do_generate() -> Any:
            with torch.no_grad():
                return cast(Callable[..., torch.Tensor], model.generate)(
                    **generate_kwargs
                )

        try:
            outputs = await asyncio.to_thread(_do_generate)
        except RuntimeError:
            logger.exception("Generation failed")
            yield LLMResponseFinal(original=None, text="")
            return

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - request_start) * 1000

        if text:
            yield LLMResponseDelta(
                content_index=None,
                item_id=response_id,
                output_index=0,
                sequence_number=0,
                delta=text,
                is_first_chunk=True,
                time_to_first_token_ms=None,
            )
        yield LLMResponseFinal(
            original=outputs,
            text=text,
            item_id=response_id,
            latency_ms=latency_ms,
            model=self.model_id,
        )

    def _build_messages(self) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        if self._conversation:
            for msg in self._conversation.messages:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def _convert_tools_to_provider_format(
        self, tools: list[ToolSchema]
    ) -> list[dict[str, Any]]:
        return convert_tools_to_transformers_format(tools)

    def unload(self) -> None:
        logger.info(f"Unloading model: {self.model_id}")
        if self._resources is not None:
            del self._resources.model
            del self._resources.tokenizer
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
