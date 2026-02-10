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

from __future__ import annotations

import asyncio
import gc
import json
import logging
import re
import time
import uuid
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

import torch

from vision_agents.core.llm.events import (
    LLMRequestStartedEvent,
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.llm_types import NormalizedToolCallItem, ToolSchema
from vision_agents.core.warmup import Warmable

from . import events

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from vision_agents.core.processors import Processor

logger = logging.getLogger(__name__)

PLUGIN_NAME = "transformers_llm"

# ---------------------------------------------------------------------------
# Shared helpers (imported by transformers_vlm.py)
# ---------------------------------------------------------------------------

DeviceType = Literal["auto", "cuda", "mps", "cpu"]
QuantizationType = Literal["none", "4bit", "8bit"]


def resolve_torch_dtype(
    config: Literal["auto", "float16", "bfloat16", "float32"],
) -> torch.dtype:
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

    from transformers import BitsAndBytesConfig

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


# ---------------------------------------------------------------------------
# Resource container
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TransformersLLM
# ---------------------------------------------------------------------------


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
    """

    def __init__(
        self,
        model: str,
        device: DeviceType = "auto",
        quantization: QuantizationType = "none",
        torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto",
        trust_remote_code: bool = False,
        max_new_tokens: int = 512,
    ):
        LLM.__init__(self)

        self.model_id = model
        self._device_config = device
        self._quantization = quantization
        self._torch_dtype_config = torch_dtype
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens

        self._resources: Optional[ModelResources] = None

        self.events.register_events_from_module(events)

    # ------------------------------------------------------------------
    # Warmable interface
    # ------------------------------------------------------------------

    async def on_warmup(self) -> ModelResources:
        logger.info(f"Loading model: {self.model_id}")
        resources = await asyncio.to_thread(self._load_model_sync)
        logger.info(f"Model loaded on device: {resources.device}")
        return resources

    def on_warmed_up(self, resource: ModelResources) -> None:
        self._resources = resource

    def _load_model_sync(self) -> ModelResources:
        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        if self._device_config == "mps":
            model = model.to(torch.device("mps"))  # type: ignore[arg-type]

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

        if participant is None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        messages = self._build_messages()
        return await self.create_response(messages=messages, stream=True)

    async def create_response(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = True,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> LLMResponseEvent:
        if self._resources is None:
            logger.error("Model not loaded. Ensure warmup() was called.")
            return LLMResponseEvent(original=None, text="")

        if messages is None:
            messages = self._build_messages()

        model = self._resources.model
        tokenizer = self._resources.tokenizer
        device = self._resources.device

        # Prepare tools if any are registered
        tools_param: Optional[List[Dict[str, Any]]] = None
        tools_spec = self.get_available_functions()
        if tools_spec:
            tools_param = self._convert_tools_to_provider_format(tools_spec)

        # Apply chat template
        template_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if tools_param:
            template_kwargs["tools"] = tools_param

        try:
            inputs = cast(
                Dict[str, Any],
                tokenizer.apply_chat_template(messages, **template_kwargs),
            )
        except Exception as e:
            if tools_param:
                logger.warning(
                    f"apply_chat_template failed with tools, retrying without: {e}"
                )
                template_kwargs.pop("tools", None)
                inputs = cast(
                    Dict[str, Any],
                    tokenizer.apply_chat_template(messages, **template_kwargs),
                )
                tools_param = None
            else:
                logger.error(f"Failed to apply chat template: {e}")
                return LLMResponseEvent(original=None, text="")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        max_tokens = max_new_tokens or self._max_new_tokens

        self.events.send(
            LLMRequestStartedEvent(
                plugin_name=PLUGIN_NAME,
                model=self.model_id,
                streaming=stream,
            )
        )

        if stream:
            result = await self._generate_streaming(
                model, tokenizer, inputs, max_tokens, temperature, do_sample
            )
        else:
            result = await self._generate_non_streaming(
                model, tokenizer, inputs, max_tokens, temperature, do_sample
            )

        # Check for tool calls in generated text
        if tools_param and result.text:
            tool_calls = self._extract_tool_calls_from_text(result.text)
            if tool_calls:
                return await self._handle_tool_calls(
                    tool_calls, messages, tools_param, kwargs
                )

        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def _generate_streaming(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> LLMResponseEvent:
        from transformers import TextStreamer

        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        class _AsyncBridgeStreamer(TextStreamer):
            """Bridges token text to an ``asyncio.Queue`` without blocking the event loop."""

            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                loop.call_soon_threadsafe(async_queue.put_nowait, text)
                if stream_end:
                    loop.call_soon_threadsafe(async_queue.put_nowait, None)

        streamer = _AsyncBridgeStreamer(
            tokenizer,  # type: ignore[arg-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
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
                    model.generate(**generate_kwargs)  # type: ignore[operator]
            except Exception as e:
                generation_error = e
                logger.error(f"Generation failed: {e}")
                # Unblock the async consumer so it doesn't hang forever
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
                    plugin_name=PLUGIN_NAME,
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
                    plugin_name=PLUGIN_NAME,
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
                plugin_name=PLUGIN_NAME,
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
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        inputs: Dict[str, Any],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> LLMResponseEvent:
        request_start = time.perf_counter()
        response_id = str(uuid.uuid4())

        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 1.0,
            "pad_token_id": tokenizer.pad_token_id,
        }

        def _do_generate() -> Any:
            with torch.no_grad():
                return model.generate(**generate_kwargs)  # type: ignore[operator]

        try:
            outputs = await asyncio.to_thread(_do_generate)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.events.send(
                events.LLMErrorEvent(
                    plugin_name=PLUGIN_NAME,
                    error_message=str(e),
                    event_data=e,
                )
            )
            return LLMResponseEvent(original=None, text="")

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency_ms = (time.perf_counter() - request_start) * 1000

        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name=PLUGIN_NAME,
                original=outputs,
                text=text,
                item_id=response_id,
                latency_ms=latency_ms,
                model=self.model_id,
            )
        )

        return LLMResponseEvent(original=outputs, text=text)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._instructions:
            messages.append({"role": "system", "content": self._instructions})
        if self._conversation:
            for msg in self._conversation.messages:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def _convert_tools_to_provider_format(
        self, tools: List[ToolSchema]
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for t in tools or []:
            name = t.get("name", "unnamed_tool")
            description = t.get("description", "") or ""
            params = t.get("parameters_schema") or t.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            params.setdefault("type", "object")
            params.setdefault("properties", {})

            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params,
                    },
                }
            )
        return result

    def _extract_tool_calls_from_text(self, text: str) -> List[NormalizedToolCallItem]:
        """Parse tool calls from raw model output text.

        Supports:
        - Hermes format: ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``
        - Generic JSON: ``{"name": ..., "arguments": ...}``
        """
        tool_calls: List[NormalizedToolCallItem] = []

        # Pattern 1: Hermes / NousResearch XML tags
        hermes_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        for match in re.finditer(hermes_pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                tool_calls.append(
                    {
                        "type": "tool_call",
                        "id": data.get("id", str(uuid.uuid4())),
                        "name": data.get("name", ""),
                        "arguments_json": data.get("arguments", {}),
                    }
                )
            except json.JSONDecodeError:
                continue

        if tool_calls:
            return tool_calls

        # Pattern 2: generic JSON objects with name + arguments keys
        json_pattern = (
            r"\{[^{}]*\"name\"\s*:[^{}]*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}"
        )
        for match in re.finditer(json_pattern, text):
            try:
                data = json.loads(match.group(0))
                if "name" in data and "arguments" in data:
                    tool_calls.append(
                        {
                            "type": "tool_call",
                            "id": str(uuid.uuid4()),
                            "name": data["name"],
                            "arguments_json": data["arguments"],
                        }
                    )
            except json.JSONDecodeError:
                continue

        return tool_calls

    async def _handle_tool_calls(
        self,
        tool_calls: List[NormalizedToolCallItem],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        kwargs: Dict[str, Any],
    ) -> LLMResponseEvent:
        """Execute tool calls and generate follow-up responses.

        Mirrors ``HuggingFaceLLM._handle_tool_calls`` using the base class
        ``_dedup_and_execute`` infrastructure.
        """
        llm_response: LLMResponseEvent = LLMResponseEvent(original=None, text="")
        max_rounds = 3
        current_tool_calls = tool_calls
        seen: set[tuple] = set()
        current_messages = list(messages)

        for round_num in range(max_rounds):
            triples, seen = await self._dedup_and_execute(
                current_tool_calls,
                max_concurrency=8,
                timeout_s=30,
                seen=seen,
            )

            if not triples:
                break

            assistant_tool_calls = []
            tool_results = []
            for tc, res, err in triples:
                cid = tc.get("id")
                if not cid:
                    continue

                assistant_tool_calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc["name"],
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

            if not tool_results:
                return llm_response

            current_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls,
                }
            )
            current_messages.extend(tool_results)

            # Follow-up generation (non-streaming during tool loops)
            follow_up = await self.create_response(
                messages=current_messages,
                stream=False,
                **kwargs,
            )

            next_tool_calls = self._extract_tool_calls_from_text(follow_up.text)
            if next_tool_calls and round_num < max_rounds - 1:
                current_tool_calls = next_tool_calls
                llm_response = follow_up
                continue

            return follow_up

        return llm_response

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

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
