import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

from ._backend import Backend
from ._generated.api.default import close_session, create_session
from ._generated.models import CreateSessionRequest, Error, Session
from ._socket import Socket

logger = logging.getLogger(__name__)


@dataclass
class TextEvent:
    """Something the backend did on its way to an answer.

    Attributes:
        type: One of ``delta``, ``answer``, ``looked_up``, ``delegated``, ``settled``
            or ``error``.
        text: The words, for a delta, an answer or a settled piece of delegated work.
        skill: Which skill was handed the work, for ``delegated`` and ``settled``.
        query: What was searched for, for ``looked_up``.
        documents: How many passages bore on it, for ``looked_up``.
        error: What went wrong, for ``error``.
    """

    type: str
    text: str = ""
    skill: str = ""
    query: str = ""
    documents: int = 0
    error: str = ""


class TextSession:
    """A conversation with the acceleration backend, held in writing.

    The backend runs the same agent a phone call would get: the same harness, the same
    skills handed to a slower model, and the same knowledge base looked up mid-answer.
    What is missing is only the voice, so there is no call to join, nothing is transcribed
    and nothing is spoken.

    Example:
        ```python
        async with stream.TextSession(config_id="docs-agent") as session:
            async for event in session.ask("how does routing fail over?"):
                if event.type == "delta":
                    print(event.text, end="", flush=True)
        ```
    """

    def __init__(
        self,
        config_id: str = "",
        instructions: str = "",
        model: str = "",
        subagent: str = "",
        skills: Optional[list[str]] = None,
        max_tokens: int = 0,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Configure a conversation to hold in writing.

        Everything named here overrides what the agent config says, so a caller can reuse
        a stored configuration and still change one thing about this conversation.

        Args:
            config_id: A stored agent config to start from. A knowledge base is only
                configured there, so a conversation that looks things up names one.
            instructions: The system prompt.
            model: The model that answers, as a `provider/model` name or a capability
                shortcut such as `llm-fast`.
            subagent: The slower model that runs the work a skill hands over. Without one
                the model answers everything itself and skills mean nothing.
            skills: Which skills the model may hand work to, by name.
            max_tokens: A ceiling on a reply. Zero leaves the backend's default.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        self.backend = Backend(url=url, customer_id=customer_id)
        self.config_id = config_id
        self.instructions = instructions
        self.model = model
        self.subagent = subagent
        self.skills = skills
        self.max_tokens = max_tokens

        self.session: Optional[Session] = None
        self._socket: Optional[Socket] = None

    async def start(self) -> Session:
        """Create the session and open the socket its answers arrive on."""
        created = await create_session.asyncio(
            client=self.backend.client(), body=self._request()
        )
        if isinstance(created, Error):
            raise RuntimeError(created.error)
        if created is None:
            raise RuntimeError("the router did not answer with a session")

        self.session = created
        self._socket = Socket(
            self.backend.socket(f"/v1/agents/sessions/{created.id}/events"),
            self.backend.headers,
        )
        await self._socket.connect()
        logger.info("holding session %s in writing", created.id)
        return created

    async def ask(self, text: str) -> AsyncIterator[TextEvent]:
        """Ask something, yielding what the backend does until it has finished answering.

        A question handed to a skill is answered over two turns: the model says something
        while the work runs, and the answer arrives once it comes back. Both are yielded,
        and the stream ends only when nothing is still being worked on.
        """
        if self._socket is None or not self._socket.open:
            raise RuntimeError("the session is not open")

        await self._socket.send({"type": "respond", "text": text})

        # Work handed to a skill outlives the turn that asked for it, so the answer is
        # not finished until nothing is still out with the subagent.
        pending: set[str] = set()
        async for frame in self._socket.frames():
            if isinstance(frame, bytes):
                continue

            kind = frame.get("type", "")
            if kind == "response_delta":
                yield TextEvent(type="delta", text=frame.get("text", ""))
            elif kind == "looked_up":
                yield TextEvent(
                    type="looked_up",
                    query=frame.get("query", ""),
                    documents=int(frame.get("documents", 0)),
                )
            elif kind == "delegated":
                pending.add(frame.get("task_id", ""))
                yield TextEvent(
                    type="delegated",
                    skill=frame.get("skill", ""),
                    text=frame.get("prompt", ""),
                )
            elif kind in ("task_settled", "task_cancelled"):
                pending.discard(frame.get("task_id", ""))
                yield TextEvent(
                    type="settled",
                    skill=frame.get("skill", ""),
                    text=frame.get("text", ""),
                    error=frame.get("error", ""),
                )
            elif kind == "error":
                yield TextEvent(type="error", error=frame.get("error", ""))
            elif kind == "responded":
                yield TextEvent(type="answer", text=frame.get("text", ""))
                if not pending:
                    return
            elif kind == "left":
                return

    async def close(self) -> None:
        """End the conversation. Safe to call after it has already ended."""
        session = self.session
        self.session = None
        if session is None:
            return

        if self._socket is not None and self._socket.open:
            await self._socket.send({"type": "close"})
        else:
            await close_session.asyncio_detailed(
                session.id, client=self.backend.client()
            )
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def __aenter__(self) -> "TextSession":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _request(self) -> CreateSessionRequest:
        """Render this conversation as a session to create."""
        request = CreateSessionRequest(text=True)
        if self.config_id:
            request.config_id = self.config_id
        if self.instructions:
            request.instructions = self.instructions
        if self.model:
            request.llm = self.model
        if self.subagent:
            request.subagent = self.subagent
        if self.skills:
            request.skill_names = self.skills
        if self.max_tokens:
            request.max_tokens = self.max_tokens
        return request
