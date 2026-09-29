"""LocateAnything-3B behind an OpenAI-compatible chat completions endpoint.

Baseten serves `chat_completions` at /v1/chat/completions, so the router's openaicompat
client reaches this deployment the way it reaches any vLLM one. The last user message
carries one `image_url` part (a data URI or an http URL) and the text query, for example
"Locate all the instances that matches the following description: car</c>truck." The
answer is the model's own text: <ref>car</ref><box><x1><y1><x2><y2></box>... with
coordinates on a 0 to 1000 grid.

The model decodes a whole answer at once rather than token by token, so a streamed
response is one content chunk followed by usage, which is all a client needs to bill it.
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
import urllib.request
import uuid
from collections.abc import AsyncIterator

import fastapi
import torch
from fastapi.responses import StreamingResponse
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_DIR = "/app/model_cache/locate-anything-3b"
MODEL_NAME = "LocateAnything-3B"
FETCH_TIMEOUT_S = 30


class BadRequest(ValueError):
    """The request cannot be answered as sent."""


class Model:
    """Loads LocateAnything once and answers chat completions with boxes."""

    def __init__(self, **kwargs) -> None:
        self._lazy_data_resolver = kwargs["lazy_data_resolver"]
        self._generation_mode = os.environ.get("GENERATION_MODE", "hybrid")
        self._max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "8192"))
        self._lock = asyncio.Lock()
        self._tokenizer = None
        self._processor = None
        self._model = None

    def load(self) -> None:
        self._lazy_data_resolver.block_until_download_complete()
        self._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                trust_remote_code=True,
            )
            .to("cuda")
            .eval()
        )
        logger.info("LocateAnything loaded, generation mode %s", self._generation_mode)

    async def predict(self, model_input: dict) -> dict | StreamingResponse:
        return await self._complete(model_input)

    async def chat_completions(
        self, model_input: dict, request: fastapi.Request
    ) -> dict | StreamingResponse:
        return await self._complete(model_input)

    async def _complete(self, model_input: dict) -> dict | StreamingResponse:
        image, query = _image_and_query(model_input.get("messages", []))
        temperature = float(model_input.get("temperature", 0.7))
        max_new_tokens = int(
            model_input.get("max_completion_tokens")
            or model_input.get("max_tokens")
            or self._max_new_tokens
        )
        mode = model_input.get("generation_mode", self._generation_mode)

        async with self._lock:
            answer, prompt_tokens = await asyncio.to_thread(
                self._generate, image, query, temperature, max_new_tokens, mode
            )
        completion_tokens = len(self._tokenizer(answer).input_ids)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())

        if not model_input.get("stream"):
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }

        async def events() -> AsyncIterator[str]:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
            }
            yield _sse(
                chunk
                | {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
            yield _sse(chunk | {"choices": [], "usage": usage})
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @torch.no_grad()
    def _generate(
        self,
        image: Image.Image,
        query: str,
        temperature: float,
        max_new_tokens: int,
        mode: str,
    ) -> tuple[str, int]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        text = self._processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._processor.process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to("cuda")

        sampling = temperature > 0
        response = self._model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self._tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=mode,
            temperature=temperature if sampling else 1.0,
            do_sample=sampling,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        answer = response[0] if isinstance(response, tuple) else response
        return answer, int(inputs["input_ids"].shape[1])


def _image_and_query(messages: list[dict]) -> tuple[Image.Image, str]:
    """Returns the image and text of the last user message.

    Args:
        messages: The chat completions messages.

    Returns:
        The decoded RGB image and the text query.

    Raises:
        BadRequest: When the last user message has no image or no text.
    """
    users = [message for message in messages if message.get("role") == "user"]
    if not users:
        raise BadRequest("a request needs a user message")
    content = users[-1].get("content")
    if not isinstance(content, list):
        raise BadRequest("the user message needs an image_url part and a text part")

    url = ""
    texts = []
    for part in content:
        if part.get("type") == "image_url" and not url:
            url = part["image_url"]["url"]
        elif part.get("type") == "text":
            texts.append(part["text"])
    if not url:
        raise BadRequest("the user message has no image_url part")
    query = " ".join(texts).strip()
    if not query:
        raise BadRequest("the user message has no text query")
    return _load_image(url), query


def _load_image(url: str) -> Image.Image:
    """Decodes a data URI or fetches an http URL.

    Args:
        url: A data:image/...;base64 URI or an http(s) URL.

    Returns:
        The image in RGB.
    """
    if url.startswith("data:"):
        _, _, encoded = url.partition(",")
        data = base64.b64decode(encoded)
    elif url.startswith(("http://", "https://")):
        # Some hosts, Wikimedia among them, refuse a request that names no user agent.
        fetch = urllib.request.Request(
            url, headers={"User-Agent": "locate-anything/1.0"}
        )
        with urllib.request.urlopen(fetch, timeout=FETCH_TIMEOUT_S) as response:
            data = response.read()
    else:
        raise BadRequest("image_url must be a data URI or an http URL")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload) + "\n\n"
