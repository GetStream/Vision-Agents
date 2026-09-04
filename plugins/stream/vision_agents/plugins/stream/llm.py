import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Optional, Union

from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import llm
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.utils import cancel_and_wait

from ._backend import Backend
from ._socket import Socket

logger = logging.getLogger(__name__)


class LLM(llm.LLM):
    """Completions routed through the acceleration backend.

    For a pipeline that stays in Python: the conversation, the turns and the speech are all
    here, and only the model is somewhere else. Failover and cost tracking work exactly as
    they do inside a session, because it is the same router picking the model.
    """

    def __init__(
        self,
        target: str = "",
        language: Optional[str] = None,
        max_tokens: int = 0,
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
        config_id: str = "",
        options: Optional[dict[str, Any]] = None,
    ):
        """Route completions to `target`.

        Args:
            target: A `provider/model` name or a capability shortcut such as `llm-fast`.
                It may instead be held in the named config.
            language: A language hint, which narrows the candidates.
            max_tokens: A ceiling on a reply. Zero leaves the backend's default.
            tags: Cost labels carried onto every request.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
            config_id: A stored router config to take the options from, by name or by id.
            options: Per-call overrides of that config's llm block. Usually built by
                `Router.llm.realtime`.
        """
        super().__init__()
        self.provider_name = "stream"
        self.model = target
        self.language = language
        self.max_tokens = max_tokens
        self.tags = tags or {}
        self.config_id = config_id
        self.options = options or {}
        self.backend = Backend(url=url, customer_id=customer_id)

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task] = None
        self._incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._answering = asyncio.Lock()

    async def start(self) -> None:
        """Open the socket and start reading completions off it."""
        self._socket = Socket(
            self.backend.socket("/v1/llm/stream"), self.backend.headers
        )
        await self._socket.connect()
        await self._socket.send(
            {
                "type": "start",
                "config_id": self.config_id,
                "target": self.model,
                "languages": [self.language] if self.language else [],
                "tags": self.tags,
                "llm": self.options,
            }
        )
        self._reader = asyncio.create_task(self._read())

    async def __aenter__(self) -> "LLM":
        """Start answering, for a caller holding the session itself."""
        await self.start()
        return self

    async def __aexit__(self, *exception) -> None:
        await self.close()

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[Union[LLMResponseDelta, LLMResponseFinal]]:
        """Answer `text` in the conversation this LLM was given."""
        if self._socket is None or not self._socket.open:
            raise RuntimeError("the completions socket is not open")

        if participant is None and self._conversation is not None:
            await self._conversation.send_message(
                role="user", user_id="user", content=text
            )

        completion = str(uuid.uuid4())
        started = time.perf_counter()

        async with self._answering:
            await self._socket.send(
                {
                    "type": "respond",
                    "id": completion,
                    "instructions": self._instructions,
                    "messages": self._messages(text),
                    "max_tokens": self.max_tokens,
                }
            )

            first = True
            while True:
                frame = await self._incoming.get()
                kind = frame.get("type", "")

                if kind == "delta":
                    elapsed = (time.perf_counter() - started) * 1000
                    yield LLMResponseDelta(
                        delta=frame.get("text", ""),
                        item_id=frame.get("id"),
                        is_first_chunk=first,
                        time_to_first_token_ms=elapsed if first else None,
                    )
                    first = False
                elif kind == "complete":
                    yield LLMResponseFinal(
                        text=frame.get("text", ""),
                        item_id=frame.get("id"),
                        latency_ms=(time.perf_counter() - started) * 1000,
                        time_to_first_token_ms=frame.get("time_to_first_token_ms"),
                        input_tokens=frame.get("input_tokens"),
                        output_tokens=frame.get("output_tokens"),
                        model=frame.get("model"),
                        original=frame,
                    )
                    return
                elif kind == "error":
                    error = RuntimeError(frame.get("error", "the completion failed"))
                    self.on_llm_error(error=error)
                    raise error
                elif kind == "closed":
                    return

    async def interrupt(self) -> None:
        """Abandon the completion in flight."""
        if self._socket is not None and self._socket.open:
            await self._socket.send({"type": "interrupt"})

    async def close(self) -> None:
        """Close the socket and stop reading it."""
        if self._reader is not None:
            await cancel_and_wait(self._reader)
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    def _messages(self, text: str) -> list[dict[str, str]]:
        """The conversation so far, as the router wants it.

        A session without a conversation keeps nothing, so what was just asked is all
        there is to send: the router refuses a response with no input rather than
        answering out of nothing.
        """
        if self._conversation is None:
            return [{"role": "user", "content": text}]
        return [
            {"role": message.role or "user", "content": message.content or ""}
            for message in self._conversation.messages
        ]

    async def _read(self) -> None:
        """Hand everything the router sends to whoever is answering."""
        if self._socket is None:
            return

        async for frame in self._socket.frames():
            if isinstance(frame, bytes):
                continue
            await self._incoming.put(frame)
