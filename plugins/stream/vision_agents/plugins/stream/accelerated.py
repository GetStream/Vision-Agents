import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import aiortc
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.harness import Harness
from vision_agents.core.llm.llm import (
    LLMResponseDelta,
    LLMResponseFinal,
    OmniLLM,
    ImageContent,
)
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
    CreateSessionRequestTags,
    Error,
    Sandbox,
    Session,
    SessionMemory,
    SessionMemoryFilter,
    SessionSkill,
    SessionTool,
    SessionToolParameters,
    SessionVideo,
    CreateSessionRequestSubagents,
)
from ._socket import Socket
from .config import ensure_agent
from .knowledge import Knowledge

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
        keyterms: Optional[list[str]] = None,
        subagents: Optional[dict[str, str]] = None,
        video_source: str = "",
        video_max_frames: int = 0,
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
            keyterms: Words the transcriber would otherwise get wrong, such as names and
                member IDs. Empty leaves whatever the stored config named.
            subagents: Named worker targets, selected by skill bindings.
            video_source: Camera or processor source for delegated capture.
            video_max_frames: Recent frames per task (1–8); zero uses configuration.
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
        self.keyterms = keyterms or []
        self.subagents = subagents or {}
        self.video_source = video_source
        self.video_max_frames = video_max_frames

        self.backend = Backend(url=url, customer_id=customer_id)
        # A knowledge base belongs to the stored config that reads it, so an agent
        # configured here rather than by name has none to fill.
        self.knowledge = Knowledge(config, self.backend)
        self.session: Optional[Session] = None

        self._socket: Optional[Socket] = None
        self._reader: Optional[asyncio.Task] = None
        self._running: set[asyncio.Task] = set()
        self._tool_tasks: dict[str, asyncio.Task] = {}
        self._events: asyncio.Queue[Optional[RemoteEvent]] = asyncio.Queue()

    @property
    def uses_video_observations(self) -> bool:
        return self.session is not None and isinstance(self.session.video, SessionVideo)

    @property
    def router_session_id(self) -> Optional[str]:
        """The session the router is running this call in, once it has joined."""
        return self.session.id if self.session else None

    async def join_remote(self, call: RemoteCall) -> None:
        """Create the session and start watching it.

        Returns once the backend is in the call, so an agent that has joined is one that
        is already listening. A call with nothing to join is held in writing instead.
        """
        request = self._request(call)
        if self.config:
            # The directory is the config, so it is stored before the config is looked
            # up: an agent whose instructions changed since the last run joins with the
            # ones on disk rather than the ones the server happens to remember.
            await ensure_agent(
                self.config,
                url=self.backend.url,
                customer_id=self.backend.customer_id,
            )
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
        logger.info(
            "joined %s remotely as session %s",
            f"call {call.call_id}" if call.call_id else "a conversation in writing",
            created.id,
        )

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

    async def respond_remote(
        self,
        text: str,
        interrupt: bool = True,
        images: Optional[list[ImageContent]] = None,
    ) -> None:
        """Answer `text` through the model, as though it had been said on the call."""
        if interrupt:
            await self._command({"type": "interrupt"})
        command: dict[str, Any] = {"type": "respond", "text": text}
        if images:
            command["images"] = [image.as_image_dict() for image in images]
        await self._command(command)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
        images: Optional[list[ImageContent]] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Answer `text` through the model.

        Yields nothing: the reply is spoken on the call and reported as events, so there
        is no response here to hand back.
        """
        await self.respond_remote(text, images=images)
        return
        yield  # pragma: no cover - the empty stream this signature promises

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
            user_id=call.agent_user_id,
            agent_id=call.agent_id or call.agent_user_id,
            backchannel=self.backchannel,
        )
        if call.call_id:
            request.call_id = call.call_id
            request.call_type = call.call_type
        else:
            # Nothing to join, so the conversation is held in writing and the agent id is
            # the channel it is written in.
            request.text = True
        # Anything named here wins over the stored config, so a field this agent does not
        # decide is left out rather than sent empty: sending it would replace what the
        # config says with nothing.
        if call.instructions:
            request.instructions = call.instructions
        if self.greeting:
            request.greeting = self.greeting
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
        if self.keyterms:
            request.keyterms = self.keyterms
        if self.video_source or self.video_max_frames:
            request.video = SessionVideo(
                source=self.video_source, max_frames=self.video_max_frames or 1
            )

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
        if self.subagent:
            request.subagent = self.subagent
        if self.subagents:
            request.subagents = CreateSessionRequestSubagents.from_dict(self.subagents)
        if harness is None:
            if self.subagent:
                request.subagent = self.subagent
            return

        spec = harness.spec()
        if "subagents" in spec:
            request.subagents = CreateSessionRequestSubagents.from_dict(
                {**self.subagents, **spec["subagents"]}
            )
        if spec["tasks"]:
            request.tasks = spec["tasks"]
        if "sandbox" in spec:
            request.sandbox = Sandbox(spec["sandbox"])
        if "skills" in spec:
            request.skills = [
                SessionSkill(
                    name=skill["name"],
                    subagent=skill["subagent"],
                    capture_video=skill["capture_video"],
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

        if kind == "tool_cancel":
            running = self._tool_tasks.get(str(frame.get("id", "")))
            if running is not None:
                running.cancel()
            return
        if kind == "tool_call":
            call_id = str(frame.get("id", ""))
            if len(self._tool_tasks) >= 16:
                await self._command(
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "error": "video worker task capacity exceeded",
                    }
                )
                return
            task = asyncio.create_task(self._run_tool(frame))
            self._running.add(task)
            self._tool_tasks[call_id] = task
            task.add_done_callback(self._running.discard)
            task.add_done_callback(lambda finished: self._tool_tasks.pop(call_id, None))
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
            if name == "get_video_frames":
                result["output"] = await self._capture_video(arguments)
            else:
                output = await self.call_function(name, arguments)
                result["output"] = _tool_output(output)
        except ValueError as exc:
            if name == "get_video_frames":
                result["output"] = str(exc)
            else:
                result["error"] = str(exc)
        except Exception as exc:
            logger.exception("the tool %s failed", name)
            result["error"] = str(exc)

        if self._socket is not None and self._socket.open:
            await self._socket.send(result)

    async def _capture_video(
        self, arguments: dict[str, Any]
    ) -> list[dict[str, object]]:
        agent = self.agent
        if agent is None:
            raise ValueError("no video worker is attached")
        buffers = [agent.observations]
        buffers.extend(
            p.observation_buffer
            for p in agent.video_processors
            if p.observation_buffer is not None
        )
        source = str(arguments.get("source", ""))
        candidates = [(name, buffer) for buffer in buffers for name in buffer.sources]
        if source:
            candidates = [
                (name, buffer)
                for name, buffer in candidates
                if name == source or name.split("/")[0] == source
            ]
        else:
            # Processor sources need an explicit choice; raw camera evidence is the default.
            candidates = [
                (name, buffer)
                for name, buffer in candidates
                if buffer is agent.observations
            ]
        if len(candidates) != 1:
            names = ", ".join(name for name, _ in candidates) or "none"
            raise ValueError(f"select one available video source (available: {names})")
        name, buffer = candidates[0]
        selected = buffer.select(
            name, int(arguments["at_ms"]), int(arguments.get("limit", 1))
        )
        parts: list[dict[str, object]] = []
        for observation in selected:
            parts.extend(await asyncio.to_thread(observation.content))
        if len(json.dumps(parts).encode()) > 4 << 20:
            raise ValueError(
                "selected images exceed the transfer limit; request fewer frames"
            )
        return parts

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
    if kind == "response_delta":
        return RemoteEvent(type="agent_speech_delta", text=frame.get("text", ""))
    if kind == "responded":
        return RemoteEvent(type="agent_speech", text=frame.get("text", ""))
    if kind == "looked_up":
        return RemoteEvent(
            type="looked_up",
            query=frame.get("query", ""),
            documents=int(frame.get("documents", 0)),
        )
    if kind == "delegated":
        return RemoteEvent(
            type="delegated",
            skill=frame.get("skill", ""),
            text=frame.get("prompt", ""),
        )
    if kind in ("task_settled", "task_cancelled"):
        return RemoteEvent(
            type="task_settled",
            skill=frame.get("skill", ""),
            text=frame.get("text", ""),
            error=frame.get("error", ""),
        )
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


def _tool_output(output: Any) -> str | list[dict[str, object]]:
    """A string when the tool returned words, or parts when it returned an image."""
    if isinstance(output, list) and all(
        isinstance(part, (str, ImageContent)) for part in output
    ):
        return [
            {"type": "text", "text": part}
            if isinstance(part, str)
            else part.as_content_part()
            for part in output
        ]
    images: list[ImageContent] = []
    rest = _take_images(output, images)
    if not images:
        return _rendered(output)
    parts: list[dict[str, object]] = []
    if rest not in (None, {}, []):
        parts.append({"type": "text", "text": _rendered(rest)})
    parts.extend(image.as_content_part() for image in images)
    return parts


def _take_images(value: Any, images: list[ImageContent]) -> Any:
    """Pull ImageContent values out of a tool result, leaving the rest."""
    if isinstance(value, ImageContent):
        images.append(value)
        return None
    if isinstance(value, dict):
        kept: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, ImageContent):
                images.append(item)
            else:
                kept[key] = _take_images(item, images)
        return kept
    if isinstance(value, list):
        kept_list: list[Any] = []
        for item in value:
            if isinstance(item, ImageContent):
                images.append(item)
            else:
                kept_list.append(_take_images(item, images))
        return kept_list
    return value
