import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import aiortc
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.harness import Harness
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal, OmniLLM
from vision_agents.core.llm.remote import (
    RemoteCall,
    RemoteEvent,
    RemotePipelineError,
)
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.utils.video_forwarder import VideoForwarder

from ._backend import Backend
from ._generated.api.default import close_session, create_session, list_agent_configs
from ._generated.models import (
    CreateSessionRequest,
    CreateSessionRequestSandbox,
    CreateSessionRequestTags,
    Error,
    Session,
    SessionMemory,
    SessionMemoryFilter,
    SessionSkill,
    SessionTool,
    SessionToolParameters,
)
from ._socket import Socket

logger = logging.getLogger(__name__)

# USER_KEY is the memory filter key naming who the memories are about. Everything else in
# the filter narrows recall; this one is what recall is keyed by.
USER_KEY = "user_id"


class Accelerated(OmniLLM):
    """A whole voice pipeline, running in the acceleration backend.

    This is an LLM by position rather than by nature: it does no inference and touches no
    media. The backend joins the call, hears the caller, answers and speaks, and what
    arrives here are the events saying so. What stays in Python is function calling, since
    the functions are here, and configuration, since the decisions are yours.

    Example:
        ```python
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=agent_user,
            llm=stream.Accelerated(model="gemma4", stt="realtime-best", tts="sonic_36"),
            harness=DefaultHarness(),
        )
        ```
    """

    def __init__(
        self,
        model: str = "",
        stt: str = "",
        tts: str = "",
        subagent: str = "",
        voice: str = "",
        config: str = "",
        language: Optional[str] = None,
        greeting: str = "",
        backchannel: bool = False,
        max_tokens: int = 0,
        tool_timeout: float = 0.0,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Configure a pipeline to run remotely.

        Every target is a `provider/model` name or a capability shortcut such as
        `llm-fast`; leaving one empty takes the backend's default for that modality.

        Args:
            model: The model that answers.
            stt: The model that transcribes.
            tts: The model that speaks.
            subagent: The model that does the thinking a harness delegates. Overridden by
                the agent's harness when it names one.
            voice: A provider-specific voice id.
            config: The name of a stored agent config to start from, as passed to
                `define_agent`. Everything else here overrides what it says. The name is
                looked up on joining, so an agent can be built before the config exists.
            language: A language hint, which narrows the candidates in every modality.
            greeting: Said on joining without going through the model. Empty means the
                agent waits to be spoken to.
            backchannel: Murmur while a caller is still talking, the way a person does.
            max_tokens: A ceiling on a reply. Zero leaves the backend's default.
            tool_timeout: How long the model waits for one of your functions before
                carrying on without it. Zero leaves the backend's default.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        super().__init__()
        self.provider_name = "stream"
        self.model = model
        self.stt = stt
        self.tts = tts
        self.subagent = subagent
        self.voice = voice
        self.config = config
        self.language = language
        self.greeting = greeting
        self.backchannel = backchannel
        self.max_tokens = max_tokens
        self.tool_timeout = tool_timeout

        self.backend = Backend(url=url, customer_id=customer_id)
        self.session: Optional[Session] = None

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task] = None
        self._running: set[asyncio.Task] = set()
        self._events: asyncio.Queue[Optional[RemoteEvent]] = asyncio.Queue()

    @property
    def router_session_id(self) -> Optional[str]:
        """The session the router is running this call in, once it has joined."""
        return self.session.id if self.session else None

    async def join_remote(self, call: RemoteCall) -> None:
        """Create the session and start watching it.

        Returns once the backend is in the call, so an agent that has joined is one that
        is already listening.
        """
        request = self._request(call)
        if self.config:
            request.config_id = await self._config_id(self.config)

        created = await create_session.asyncio(
            client=self.backend.client(), body=request
        )
        if isinstance(created, Error):
            raise RemotePipelineError(created.error)
        if created is None:
            raise RemotePipelineError("the router did not answer with a session")

        self.session = created
        # Decisions are the router explaining itself several times a second, which is what
        # a dashboard watching a call wants and what this would only throw away.
        self._socket = Socket(
            self.backend.socket(
                f"/v1/agents/sessions/{created.id}/events?decisions=false"
            ),
            self.backend.headers,
        )
        await self._socket.connect()
        self._reader = asyncio.create_task(self._watch())
        logger.info("joined call %s remotely as session %s", call.call_id, created.id)

    async def remote_events(self) -> AsyncIterator[RemoteEvent]:
        """Yield what the backend did until the call ends."""
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def say_remote(self, text: str, interrupt: bool = False) -> None:
        """Speak `text` on the call without going through the model."""
        if interrupt:
            await self._command({"type": "interrupt"})
        await self._command({"type": "say", "text": text})

    async def respond_remote(self, text: str, interrupt: bool = True) -> None:
        """Answer `text` through the model, as though it had been said on the call."""
        if interrupt:
            await self._command({"type": "interrupt"})
        await self._command({"type": "respond", "text": text})

    async def leave_remote(self) -> None:
        """End the call. Safe to call after it has already ended."""
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
        await self._stop_watching()

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Answer `text` through the model.

        Yields nothing: the reply is spoken on the call and reported as events, so there
        is no response here to hand back.
        """
        await self.respond_remote(text)
        return
        yield  # pragma: no cover - the empty stream this signature promises

    async def interrupt(self) -> None:
        """Abandon the reply being spoken."""
        await self._command({"type": "interrupt"})

    async def close(self) -> None:
        await self.leave_remote()
        await self._stop_watching()

    async def simple_audio_response(self, pcm: PcmData, participant: Participant):
        """Ignore audio. The backend is on the call and hears the caller directly."""

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Ignore video, for the same reason audio is ignored."""

    async def stop_watching_video_track(self) -> None:
        """Nothing was being watched."""

    async def _config_id(self, name: str) -> str:
        """Find the id of the stored config called `name`.

        A config is named when it is defined and identified by id everywhere after, so the
        lookup happens here rather than making the caller carry an id around.
        """
        listed = await list_agent_configs.asyncio(client=self.backend.client())
        if isinstance(listed, Error):
            raise RemotePipelineError(listed.error)
        if listed is None:
            raise RemotePipelineError(
                "the router did not answer with any agent configs"
            )

        for stored in listed:
            if stored.name == name:
                return stored.id
        raise RemotePipelineError(f"there is no agent config called {name!r}")

    def _request(self, call: RemoteCall) -> CreateSessionRequest:
        """Render the agent's configuration as a session to create."""
        request = CreateSessionRequest(
            call_id=call.call_id,
            call_type=call.call_type,
            user_id=call.agent_user_id,
            agent_id=call.agent_user_id,
            instructions=call.instructions,
            greeting=self.greeting,
            backchannel=self.backchannel,
        )
        if self.model:
            request.llm = self.model
        if self.stt:
            request.stt = self.stt
        if self.tts:
            request.tts = self.tts
        if self.voice:
            request.voice = self.voice
        if self.language:
            request.languages = [self.language]
        if self.max_tokens:
            request.max_tokens = self.max_tokens
        if self.tool_timeout:
            request.tool_timeout_ms = int(self.tool_timeout * 1000)

        tools = self._tools()
        if tools:
            request.tools = tools

        if call.cost_tracking:
            tags = CreateSessionRequestTags()
            tags.additional_properties = {
                key: str(value) for key, value in call.cost_tracking.items()
            }
            request.tags = tags

        if call.memory_filter:
            request.memory = self._memory(call.memory_filter)

        self._apply_harness(request, call.harness)
        return request

    def _tools(self) -> list[SessionTool]:
        """The functions registered here, as the model will be offered them."""
        tools = []
        for schema in self.get_available_functions():
            tool = SessionTool(
                name=schema["name"],
                description=schema.get("description", ""),
            )
            parameters = SessionToolParameters()
            parameters.additional_properties = dict(schema.get("parameters_schema", {}))
            tool.parameters = parameters
            tools.append(tool)
        return tools

    def _memory(self, memory_filter: dict[str, str]) -> SessionMemory:
        """Split the filter into who the memories are about and what narrows them."""
        memory = SessionMemory()
        if USER_KEY in memory_filter:
            memory.user_id = str(memory_filter[USER_KEY])

        narrowing = {
            key: str(value) for key, value in memory_filter.items() if key != USER_KEY
        }
        if narrowing:
            extra = SessionMemoryFilter()
            extra.additional_properties = narrowing
            memory.filter_ = extra
        return memory

    def _apply_harness(
        self, request: CreateSessionRequest, harness: Optional[Harness]
    ) -> None:
        """Fold the agent's harness into the session it is configuring."""
        if harness is None:
            if self.subagent:
                request.subagent = self.subagent
            return

        spec = harness.spec()
        request.subagent = spec.get("subagent", self.subagent)
        if spec["tasks"]:
            request.tasks = spec["tasks"]
        if "sandbox" in spec:
            request.sandbox = CreateSessionRequestSandbox(spec["sandbox"])
        if "skills" in spec:
            request.skills = [
                SessionSkill(
                    name=skill["name"],
                    description=skill["description"],
                    instructions=skill["instructions"],
                    deadline_ms=skill["deadline_ms"],
                )
                for skill in spec["skills"]
            ]

    async def _command(self, frame: dict[str, Any]) -> None:
        """Act on the session over the socket it is being watched on."""
        if self._socket is None or not self._socket.open:
            raise RemotePipelineError("the agent is not on a call")
        await self._socket.send(frame)

    async def _watch(self) -> None:
        """Read the session's socket until it ends, translating as it goes."""
        if self._socket is None:
            return

        try:
            async for frame in self._socket.frames():
                if isinstance(frame, bytes):
                    continue
                await self._received(frame)
        finally:
            await self._events.put(None)

    async def _received(self, frame: dict[str, Any]) -> None:
        """Turn one session frame into an event, or into a tool call to run."""
        kind = frame.get("type", "")

        if kind == "tool_call":
            task = asyncio.create_task(self._run_tool(frame))
            self._running.add(task)
            task.add_done_callback(self._running.discard)
            return

        event = _event_of(frame)
        if event is not None:
            await self._events.put(event)

    async def _run_tool(self, frame: dict[str, Any]) -> None:
        """Run one of the caller's functions and answer the model with what it said.

        A failure is reported rather than raised: the model is mid-sentence waiting for
        this, and it can say something useful about a tool that did not work only if it is
        told that it did not work.
        """
        call_id = frame.get("id", "")
        name = frame.get("name", "")
        result: dict[str, Any] = {"type": "tool_result", "tool_call_id": call_id}

        try:
            arguments = json.loads(frame.get("arguments") or "{}")
            output = await self.call_function(name, arguments)
            result["output"] = _rendered(output)
        except Exception as exc:
            logger.exception("the tool %s failed", name)
            result["error"] = str(exc)

        if self._socket is not None and self._socket.open:
            await self._socket.send(result)

    async def _stop_watching(self) -> None:
        """Drop the socket and everything reading it."""
        if self._reader is not None:
            await cancel_and_wait(self._reader)
            self._reader = None

        for task in list(self._running):
            await cancel_and_wait(task)
        self._running.clear()

        if self._socket is not None:
            await self._socket.close()
            self._socket = None

        await self._events.put(None)


def _event_of(frame: dict[str, Any]) -> Optional[RemoteEvent]:
    """Translate a session frame into the agent's terms.

    Most of what a conversation reports has no counterpart in the agent, which records
    speech and turns. The rest is left to whoever is watching the session directly.
    """
    kind = frame.get("type", "")
    participant = frame.get("participant") or {}

    if kind == "participant_joined":
        return RemoteEvent(
            type="participant_joined",
            user_id=participant.get("user_id", ""),
            participant_id=participant.get("id", ""),
        )
    if kind == "participant_left":
        return RemoteEvent(
            type="participant_left",
            user_id=participant.get("user_id", ""),
            participant_id=participant.get("id", ""),
        )
    if kind == "heard":
        return RemoteEvent(
            type="user_speech",
            text=frame.get("text", ""),
            user_id=participant.get("user_id", ""),
            participant_id=participant.get("id", ""),
        )
    if kind == "responding":
        return RemoteEvent(
            type="agent_turn_started",
            user_id=participant.get("user_id", ""),
            participant_id=participant.get("id", ""),
        )
    if kind == "responded":
        return RemoteEvent(type="agent_speech", text=frame.get("text", ""))
    if kind == "turn":
        return RemoteEvent(
            type="agent_turn_ended",
            interrupted=bool(frame.get("interrupted")),
            user_id=participant.get("user_id", ""),
            participant_id=participant.get("id", ""),
        )
    if kind == "error":
        return RemoteEvent(type="error", error=frame.get("error", ""))
    if kind == "left":
        return RemoteEvent(type="ended")

    logger.debug("no agent event for a %s frame", kind)
    return None


def _rendered(output: Any) -> str:
    """Render what a function returned in words the model can use."""
    if isinstance(output, str):
        return output
    return json.dumps(output)
