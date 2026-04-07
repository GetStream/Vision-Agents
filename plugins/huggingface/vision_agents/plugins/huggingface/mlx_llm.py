"""
MlxLLM - Local text LLM inference via Apple MLX.

Runs quantized models on Apple Silicon using the mlx-lm library.

Example:
    from vision_agents.plugins.huggingface import MlxLLM

    llm = MlxLLM(model="mlx-community/gemma-4-e4b-it-8bit")
"""

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Optional

from mlx_lm import generate, load, stream_generate
from mlx_lm.sample_utils import make_sampler

from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent

from . import events
from ._local_inference import LocalTextLLM

logger = logging.getLogger(__name__)


class MlxModelResources:
    """Container for a loaded MLX model and tokenizer."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer


class MlxLLM(LocalTextLLM[MlxModelResources]):
    """Local LLM inference using Apple MLX.

    Runs quantized models efficiently on Apple Silicon via the ``mlx-lm``
    library. Models from ``mlx-community`` on HuggingFace come pre-quantized.

    Args:
        model: HuggingFace model ID (e.g. ``"mlx-community/gemma-4-e4b-it-8bit"``).
        max_new_tokens: Default maximum tokens to generate per response.
        max_tool_rounds: Maximum tool-call rounds per response (default 3).
    """

    _plugin_name = "mlx_llm"

    def _load_model_sync(self) -> MlxModelResources:
        result = load(self.model_id)
        return MlxModelResources(model=result[0], tokenizer=result[1])

    def _apply_template(
        self,
        messages: list[dict[str, Any]],
        tools_param: Optional[list[dict[str, Any]]],
    ) -> tuple[Any, bool] | None:
        assert self._resources is not None
        tokenizer = self._resources.tokenizer
        template_kwargs: dict[str, Any] = {"add_generation_prompt": True}
        if tools_param:
            template_kwargs["tools"] = tools_param

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, **template_kwargs
            )
            return prompt, tools_param is not None
        except (TypeError, ValueError) as e:
            if tools_param:
                logger.warning(
                    "apply_chat_template failed with tools, retrying without: %s", e
                )
                template_kwargs.pop("tools", None)
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, **template_kwargs
                )
                return prompt, False
            logger.exception("Failed to apply chat template")
            return None

    async def _generate_streaming(
        self, prepared_input: Any, max_tokens: int, temperature: float
    ) -> LLMResponseEvent:
        assert self._resources is not None
        model = self._resources.model
        tokenizer = self._resources.tokenizer

        request_start = time.perf_counter()
        first_token_time: Optional[float] = None
        text_chunks: list[str] = []
        chunk_index = 0
        response_id = str(uuid.uuid4())
        sampler = make_sampler(temperature)

        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        generation_error: Optional[RuntimeError | ValueError] = None

        def _stream_to_queue() -> None:
            nonlocal generation_error
            try:
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt=prepared_input,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    loop.call_soon_threadsafe(async_queue.put_nowait, response.text)
            except (RuntimeError, ValueError) as e:
                generation_error = e
                logger.exception("Generation failed")
            finally:
                loop.call_soon_threadsafe(async_queue.put_nowait, None)

        thread = threading.Thread(target=_stream_to_queue, daemon=True)
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
        sampler = make_sampler(temperature)

        result = await asyncio.to_thread(
            generate,
            model,
            tokenizer,
            prompt=prepared_input,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        text = result.text if hasattr(result, "text") else str(result)

        if emit_events:
            latency_ms = (time.perf_counter() - request_start) * 1000
            self.events.send(
                LLMResponseCompletedEvent(
                    plugin_name=self._plugin_name,
                    original=None,
                    text=text,
                    item_id=response_id,
                    latency_ms=latency_ms,
                    model=self.model_id,
                )
            )

        return LLMResponseEvent(original=None, text=text)
