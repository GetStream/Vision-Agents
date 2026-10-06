from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional, Union

from vision_agents.core.llm.function_registry import FunctionRegistry

from ._backend import API_KEY_ENV, API_SECRET_ENV, Backend
from ._generated.api.default import (
    create_session,
    delete_session,
    delete_session_memories,
    fork_session,
    get_session,
    query_sessions,
    stop_session,
    update_session,
)
from ._generated.models import (
    CreateSessionRequest,
    CreateSessionRequestCustom,
    ForkSessionRequest,
    ForkSessionRequestCustom,
    ModelOverwrites,
    Session as SessionRow,
    SessionFilter,
    SessionPage,
    SessionQuery,
    SessionTool,
    SessionToolParameters,
    TextMatch,
    UpdateSessionRequest,
    UpdateSessionRequestCustom,
    UpdateSessionRequestThinking,
    UpdateSessionRequestVerbosity,
)
from ._socket import Socket
from .responses import Responses, _deleted, _unwrapped

logger = logging.getLogger(__name__)

# How many tool calls may run at once before the model is told to carry on without one. The
# same ceiling the pipeline uses, for the same reason: a model that has asked for sixteen
# things is not waiting on the seventeenth.
RUNNING_TOOLS = 16


@dataclass
class Participant:
    """Somebody on the call, as the backend reports them."""

    id: str = ""
    user_id: str = ""
    name: str = ""


@dataclass
class SessionEvent:
    """One thing the conversation did.

    ``kind`` is the backend's own name for it: joined, hearing, heard, decision, responding,
    response_delta, responded, blocked, spoke, turn, delegated, task_settled, task_cancelled,
    tool_started, tool_ran, transferred, pressed, looked_up, backchannel, interrupted,
    overlap_decided, conversation_compacted, error and left. The fields below are filled from
    whichever of those carry them, and ``frame`` is the whole thing for anything they do not
    cover.
    """

    kind: str = ""
    text: str = ""
    participant: Optional[Participant] = None
    interrupted: bool = False
    pending_work: bool = False
    error: str = ""
    frame: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionOptions:
    """What a conversation is opened with, beyond which agent holds it.

    Everything here is the caller's own decision about this one conversation: what to call it
    so they can find it again, which models to overrule, and whether to keep it at all.

    Attributes:
        id: The UUID to hold the session by, for a caller that wants to know it before the
            session exists. Empty lets the router generate one. One already taken is a 409.
        title: What a person finds the conversation by later. Searched.
        description: A longer note, searched alongside the title.
        project_id: Groups conversations, and is carried as a cost label too.
        custom: The caller's own labels, which a query can match on.
        incognito: Hold the conversation and keep nothing: no row, no turns, no transcript. It
            cannot be searched for, listed or forked afterwards, which is the point of it.
            Otherwise a text conversation is kept in Stream Chat, so it outlives the session.
        model_overwrites: What to change about the models for this conversation alone.
        call_id: The call to join. Empty holds the conversation in writing.
        call_type: The Stream call type. Empty leaves the backend's default.
        conversation_id: The channel an earlier session was held in, to resume.
        instructions: Overrides the agent's own system prompt for this conversation.
        user_id: Who the conversation belongs to, for a backend opening one on somebody's
            behalf. A client acting for a user leaves it empty: the token already says who.
        interim: Also report what the caller is part way through saying.
        decisions: Report the router's own routing decisions. Off here by default, because it
            is the router explaining itself several times a second.
    """

    id: str = ""
    title: str = ""
    description: str = ""
    project_id: str = ""
    custom: Optional[dict[str, Any]] = None
    incognito: bool = False
    model_overwrites: Optional[ModelOverwrites] = None

    call_id: str = ""
    call_type: str = ""
    conversation_id: str = ""
    instructions: str = ""
    user_id: str = ""

    interim: bool = False
    decisions: bool = False


@dataclass
class ForkOptions:
    """What to change about a conversation while continuing it.

    Attributes:
        agent: Continue with a different agent, which is one of the reasons to fork.
        call_id: The call the fork joins, required when the parent held one and refused when
            it did not: a voice conversation cannot be forked into a written one.
        messages: Carry the parent's history across. False starts the same configuration over
            from nothing, which is what comparing two answers to one opening question wants.
        response_id: Carry the history only up to the end of this response, so the fork
            branches from that point rather than from where the parent is now.
    """

    agent: str = ""
    title: str = ""
    description: str = ""
    project_id: str = ""
    custom: Optional[dict[str, Any]] = None
    instructions: str = ""
    incognito: bool = False
    model_overwrites: Optional[ModelOverwrites] = None
    call_id: str = ""
    messages: bool = True
    response_id: str = ""

    interim: bool = False
    decisions: bool = False


@dataclass
class Query:
    """Which of an agent's conversations to list.

    Attributes:
        project_id: Only this project's. A search covers every project, so it refuses one.
        user_id: Only this user's, which only a server-side caller may ask for: anybody else
            is narrowed to their own whatever they send.
        modality: How the user took part: ``text``, ``voice`` or ``video``.
        state: ``live`` or ``ended``. Empty is both.
        agent_id: Only the sessions created with this agent id.
        limit: Up to 200. Zero is 25.
        cursor: The ``next_cursor`` of the page before, with the same filters. Empty is the
            first page.
    """

    project_id: str = ""
    user_id: str = ""
    modality: str = ""
    state: str = ""
    agent_id: str = ""
    limit: int = 0
    cursor: str = ""


class Sessions:
    """One agent's conversations: the ones being held and the ones that were."""

    def __init__(self, backend: Backend, agent: str, functions: FunctionRegistry):
        self._backend = backend
        self._agent = agent
        self._functions = functions

    async def create(self, options: Optional[SessionOptions] = None) -> "Session":
        """Open a conversation and start watching it.

        Returns once the backend is holding the conversation, so a session that has opened is
        one that is already listening. Without a ``call_id`` it is held in writing, which is
        what a conversation somebody comes back to usually is.
        """
        options = options or SessionOptions()
        created = await create_session.asyncio(
            client=self._backend.client(), body=self._request(options)
        )
        row = _unwrapped(created, f"opening a session with {self._agent}")
        return await Session.watching(self._backend, row, self._functions, options)

    async def query(self, query: Optional[Query] = None) -> SessionPage:
        """A page of the agent's conversations, most recently updated first, the ones that
        ended included. Pass the page's ``next_cursor`` as ``Query.cursor`` for the next one.

        What comes back are the rows rather than live handles: reading a conversation back is
        not the same as holding one, and most of these are over.
        """
        listed = await query_sessions.asyncio(
            client=self._backend.client(), body=self._query("", query or Query())
        )
        return _unwrapped(listed, f"listing the sessions of {self._agent}")

    async def search(self, text: str, query: Optional[Query] = None) -> SessionPage:
        """Find a conversation by what it was called, best match first.

        It reads the title, the description and the opening question, which is what a person
        remembers a conversation by. An incognito conversation is never found: nothing about
        it was written down to search. It pages the same way ``query`` does.
        """
        found = await query_sessions.asyncio(
            client=self._backend.client(), body=self._query(text, query or Query())
        )
        return _unwrapped(found, f"searching the sessions of {self._agent}")

    async def get(self, id: str) -> SessionRow:
        """One conversation, whether or not it is still being held."""
        got = await get_session.asyncio(id, client=self._backend.client())
        return _unwrapped(got, f"reading the session {id}")

    async def update(
        self,
        id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        custom: Optional[dict[str, Any]] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        stt: Optional[str] = None,
        tts: Optional[str] = None,
        sts: Optional[str] = None,
        voice: Optional[str] = None,
        thinking: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> SessionRow:
        """Change one conversation, whether or not it is still being held.

        Server side only. A field left as None is left as it is; an empty ``sts`` makes the
        session a cascade again, and an empty ``voice`` returns to the provider's default.
        One that ended can still be renamed and relabelled; instructions, models and voice
        need it running, and take over from its next turn. The agent config it started from
        is untouched.

        Args:
            thinking: ``none``, ``minimal``, ``low``, ``medium`` or ``high``.
            verbosity: ``low``, ``medium`` or ``high``.
        """
        request = UpdateSessionRequest()
        for name, value in (
            ("title", title),
            ("description", description),
            ("instructions", instructions),
            ("llm", llm),
            ("stt", stt),
            ("tts", tts),
            ("sts", sts),
            ("voice", voice),
            ("temperature", temperature),
            ("max_output_tokens", max_output_tokens),
        ):
            if value is not None:
                setattr(request, name, value)
        if custom is not None:
            request.custom = UpdateSessionRequestCustom.from_dict(custom)
        if thinking is not None:
            request.thinking = UpdateSessionRequestThinking(thinking)
        if verbosity is not None:
            request.verbosity = UpdateSessionRequestVerbosity(verbosity)

        updated = await update_session.asyncio(
            id, client=self._backend.client(), body=request
        )
        return _unwrapped(updated, f"updating the session {id}")

    async def delete(self, id: str) -> None:
        """Delete a conversation, running or ended: it is stopped, and its turns and what it
        remembered are deleted with it. The user's other memories are kept."""
        deleted = await delete_session.asyncio_detailed(
            id, client=self._backend.client()
        )
        _deleted(deleted, f"deleting the session {id}")

    async def delete_memories(self, id: str) -> None:
        """Delete what one conversation remembered, running or ended, and leave the rest of
        the user's memories alone. Server side only."""
        deleted = await delete_session_memories.asyncio_detailed(
            id, client=self._backend.client()
        )
        _deleted(deleted, f"deleting the memories of {id}")

    def responses(self, id: str) -> Responses:
        """A session's turns, read back without holding the conversation.

        For a conversation that has ended, or one being held somewhere else: the rows are in
        the backend either way, so reading them needs no socket.
        """
        return Responses(self._backend, id)

    def _request(self, options: SessionOptions) -> CreateSessionRequest:
        """Render the options as the session request, with the agent named by name."""
        request = CreateSessionRequest(agent=self._agent)
        for name in (
            "id",
            "title",
            "description",
            "project_id",
            "instructions",
            "call_type",
        ):
            if getattr(options, name):
                setattr(request, name, getattr(options, name))
        if options.conversation_id:
            request.conversation_id = options.conversation_id
        if options.user_id:
            request.user_id = options.user_id
        if options.model_overwrites is not None:
            request.model_overwrites = options.model_overwrites
        if options.custom:
            request.custom = CreateSessionRequestCustom.from_dict(options.custom)

        if options.call_id:
            request.call_id = options.call_id
        else:
            # Held in writing unless a call was named, which is what this surface is mostly
            # for: a conversation somebody comes back to.
            request.text = True

        if options.incognito:
            request.incognito = True

        declared = _tools(self._functions)
        if declared:
            request.tools = declared
        return request

    def _query(self, text: str, query: Query) -> SessionQuery:
        """The query the listing and the search share, narrowed to this agent. Text makes
        it a search.

        One function rather than two, so the two cannot drift apart in what they honour: a
        filter respected by one and forgotten by the other would be a surprise at best, and at
        worst a list somebody reads another user's conversations out of.
        """
        narrowed = SessionFilter(agent=self._agent)
        for name in ("project_id", "user_id", "modality", "state", "agent_id"):
            if getattr(query, name):
                setattr(narrowed, name, getattr(query, name))
        if text:
            narrowed.text = TextMatch(q=text)

        body = SessionQuery(filter_=narrowed)
        if query.limit:
            body.limit = query.limit
        if query.cursor:
            body.cursor = query.cursor
        return body


class Session:
    """One conversation, held in the acceleration backend.

    Nothing here does inference or touches media. The backend hears the caller, answers and
    speaks, and what arrives here are the events saying so. What stays here is function
    calling, because the functions are here.
    """

    def __init__(
        self,
        backend: Backend,
        created: SessionRow,
        functions: FunctionRegistry,
        socket: Socket,
    ):
        self.created = created
        self.responses = Responses(
            backend, created.id, kept=bool(created.conversation_id)
        )

        self._backend = backend
        self._functions = functions
        self._socket = socket
        self._events: asyncio.Queue[Optional[SessionEvent]] = asyncio.Queue()
        self._running: dict[str, asyncio.Task] = {}
        self._reader = asyncio.create_task(self._watch())
        self._ended = False

    @classmethod
    async def watching(
        cls,
        backend: Backend,
        created: SessionRow,
        functions: FunctionRegistry,
        options: Union[SessionOptions, ForkOptions],
    ) -> "Session":
        """Start watching a session the router has already created.

        A classmethod because a fork is created by a different request and is otherwise the
        same thing afterwards: one socket, one tool runner, one way of being closed.
        """
        query = []
        if options.interim:
            query.append("interim=true")
        if not options.decisions:
            query.append("decisions=false")
        path = f"/v1/agents/sessions/{created.id}/events"
        if query:
            path += "?" + "&".join(query)

        socket = Socket(backend.socket(path), backend.headers)
        try:
            await socket.connect()
        except Exception:
            # The session is live in the backend even though nothing here can watch it, so it
            # is stopped rather than left holding a call nobody is listening to.
            with contextlib.suppress(Exception):
                await stop_session.asyncio_detailed(created.id, client=backend.client())
            raise
        return cls(backend, created, functions, socket)

    @property
    def id(self) -> str:
        """The backend's id for the conversation."""
        return self.created.id

    @property
    def conversation_id(self) -> str:
        """The channel replies are written into, empty for one that keeps none."""
        return str(self.created.conversation_id or "")

    @property
    def live(self) -> bool:
        """Whether the conversation is still being held."""
        return not self._ended

    @property
    def functions(self) -> FunctionRegistry:
        """The functions this conversation offers the model, to register into."""
        return self._functions

    async def events(self) -> AsyncIterator[SessionEvent]:
        """Yield what the backend did until the conversation ends.

        There is one stream: two loops over it would take half the events each.
        """
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def say(self, text: str, interrupt: bool = False) -> None:
        """Speak text without going through the model, for when you know what to say."""
        if interrupt:
            await self._command({"type": "interrupt"})
        await self._command({"type": "say", "text": text})

    async def interrupt(self) -> None:
        """Abandon the reply being spoken."""
        await self._command({"type": "interrupt"})

    async def set_instructions(self, instructions: str) -> None:
        """Change what the agent is told to be, from the next turn."""
        await self._command({"type": "instructions", "instructions": instructions})

    async def fork(self, options: Optional[ForkOptions] = None) -> "Session":
        """Continue this conversation as a new one.

        What a fork is for is asking the same question differently: from here on with a harder
        model, or of a different agent, or down a branch to be kept apart from the one already
        there. The parent is untouched and keeps its own transcript; the fork writes its own,
        so continuing a conversation twice gives two readable transcripts rather than one with
        both halves interleaved. An incognito conversation cannot be forked, because there is
        nothing to fork from.

        The fork inherits this session's functions: they are here in this process, and a
        conversation continued without them would offer the model tools nothing can run.
        """
        options = options or ForkOptions()
        request = ForkSessionRequest()
        for name in (
            "agent",
            "title",
            "description",
            "project_id",
            "instructions",
            "call_id",
            "response_id",
        ):
            if getattr(options, name):
                setattr(request, name, getattr(options, name))
        if options.incognito:
            request.incognito = True
        if options.model_overwrites is not None:
            request.model_overwrites = options.model_overwrites
        if options.custom:
            request.custom = ForkSessionRequestCustom.from_dict(options.custom)
        # The generated request carries the router's own default, so this is said either way
        # rather than left out: an absent field and a true one mean the same thing.
        request.messages = options.messages

        forked = await fork_session.asyncio(
            self.id, client=self._backend.client(), body=request
        )
        row = _unwrapped(forked, f"forking the session {self.id}")
        return await Session.watching(self._backend, row, self._functions, options)

    async def update(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        custom: Optional[dict[str, Any]] = None,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        stt: Optional[str] = None,
        tts: Optional[str] = None,
        sts: Optional[str] = None,
        voice: Optional[str] = None,
        thinking: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        verbosity: Optional[str] = None,
    ) -> SessionRow:
        """Change this conversation: its title, description, custom labels, instructions,
        models or voice. See ``Sessions.update``."""
        return await self._sessions().update(
            self.id,
            title=title,
            description=description,
            custom=custom,
            instructions=instructions,
            llm=llm,
            stt=stt,
            tts=tts,
            sts=sts,
            voice=voice,
            thinking=thinking,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            verbosity=verbosity,
        )

    async def delete(self) -> None:
        """Delete this conversation. See ``Sessions.delete``."""
        await self._sessions().delete(self.id)

    async def delete_memories(self) -> None:
        """Delete what this conversation remembered. See ``Sessions.delete_memories``."""
        await self._sessions().delete_memories(self.id)

    def chat(self):
        """The Stream Chat channel this conversation is written into.

        It needs a credential of its own: the channel is Stream Chat rather than this router,
        so a client reached by customer id has nothing to connect with. A conversation that
        keeps no transcript has no channel, and an incognito one never does.
        """
        channel = self.conversation_id
        if not channel:
            raise ValueError(
                f"the session {self.id} keeps no transcript, so there is no channel to read: "
                "only a text session has one, and an incognito session never does"
            )

        client = self._stream()
        # The wire writes a conversation as type:id, which is what a Stream Chat CID is.
        # Splitting it here keeps that spelling out of the caller's way.
        kind, _, name = channel.partition(":")
        if not name:
            kind, name = "agent", channel
        return client.chat.channel(kind, name)

    def video(self):
        """The Stream call the agent is on, ready to be joined.

        A conversation held in writing joins no call, and asking for one says so rather than
        handing back a call nobody is in.
        """
        if not self.created.call_id:
            raise ValueError(
                f"the session {self.id} is held in writing, so there is no call to join"
            )
        client = self._stream()
        return client.video.call(
            str(self.created.call_type or "agent"), str(self.created.call_id)
        )

    async def close(self) -> None:
        """Stop the conversation. Safe to call after it has already ended. What it recorded
        and remembered is kept; ``delete`` takes it away."""
        if self._socket.open:
            await self._socket.send({"type": "close"})
        elif not self._ended:
            with contextlib.suppress(Exception):
                await stop_session.asyncio_detailed(
                    self.id, client=self._backend.client()
                )

        await self._socket.close()
        for task in list(self._running.values()):
            task.cancel()
        if not self._reader.done():
            self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reader
        self._ended = True
        await self._events.put(None)

    def _stream(self):
        """The Stream client, for the chat and video this conversation is held in.

        A server credential rather than this backend's token, because the ``getstream``
        package signs its own requests and has no user-token mode. That is the right shape for
        Python, which is a backend: it is the browser SDKs that connect as a user, with the
        token ``Backend.stream_credentials`` hands out.
        """
        key = self._backend.api_key or os.environ.get(API_KEY_ENV, "")
        secret = self._backend.api_secret or os.environ.get(API_SECRET_ENV, "")
        if not key or not secret:
            raise ValueError(
                "chat and video connect to Stream rather than to this router, so they need "
                f"{API_KEY_ENV} and {API_SECRET_ENV}"
            )

        from getstream import Stream

        return Stream(api_key=str(key), api_secret=str(secret))

    def _sessions(self) -> Sessions:
        """The resource methods that act on a session by id, which never need its agent."""
        return Sessions(self._backend, "", self._functions)

    async def _command(self, frame: dict[str, Any]) -> None:
        """Act on the session over the socket it is being watched on."""
        if not self._socket.open:
            raise RuntimeError(f"the session {self.id} is not being held")
        await self._socket.send(frame)

    async def _watch(self) -> None:
        """Read the socket until the conversation ends, answering tool calls as they arrive.

        It runs whether or not anybody is reading events, because a tool call the model is
        waiting on cannot depend on the caller having started a loop.
        """
        try:
            async for frame in self._socket.frames():
                if isinstance(frame, bytes):
                    continue
                await self._received(frame)
        finally:
            self._ended = True
            await self._events.put(None)

    async def _received(self, frame: dict[str, Any]) -> None:
        kind = str(frame.get("type", ""))
        call_id = str(frame.get("id", ""))

        if kind == "tool_cancel":
            running = self._running.get(call_id)
            if running is not None:
                running.cancel()
            return
        if kind == "tool_call":
            if len(self._running) >= RUNNING_TOOLS:
                await self._command(
                    {
                        "type": "tool_result",
                        "tool_call_id": call_id,
                        "error": "too many tools are already running",
                    }
                )
                return
            task = asyncio.create_task(self._run_tool(frame))
            self._running[call_id] = task
            task.add_done_callback(lambda _: self._running.pop(call_id, None))
            return

        await self._events.put(_event_of(frame))

    async def _run_tool(self, frame: dict[str, Any]) -> None:
        """Run one of the caller's functions and answer the model with what it said.

        A failure is reported rather than raised: the model is mid-sentence waiting for this,
        and it can say something useful about a tool that did not work only if it is told that
        it did not work.
        """
        name = str(frame.get("name", ""))
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": frame.get("id", ""),
        }
        # A durable command's result is only accepted back with the command and turn it names.
        for key in ("command_id", "turn_id"):
            if frame.get(key):
                result[key] = frame[key]
        try:
            arguments = json.loads(frame.get("arguments") or "{}")
            output = await self._functions.call_function(name, arguments)
            result["output"] = output if isinstance(output, str) else json.dumps(output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("the tool %s failed", name)
            result["error"] = str(exc)

        if self._socket.open:
            await self._socket.send(result)


def _tools(functions: FunctionRegistry) -> list[SessionTool]:
    """The functions registered here, as the model will be offered them."""
    declared = []
    for schema in functions.get_tool_schemas():
        tool = SessionTool(
            name=schema["name"], description=schema.get("description", "")
        )
        parameters = SessionToolParameters()
        parameters.additional_properties = dict(schema.get("parameters_schema", {}))
        tool.parameters = parameters
        declared.append(tool)
    return declared


def _event_of(frame: dict[str, Any]) -> SessionEvent:
    """Fill in the fields the frames that carry them have in common."""
    who = frame.get("participant") or {}
    return SessionEvent(
        kind=str(frame.get("type", "")),
        text=str(frame.get("text", "")),
        participant=Participant(
            id=str(who.get("id", "")),
            user_id=str(who.get("user_id", "")),
            name=str(who.get("name", "")),
        )
        if who
        else None,
        interrupted=bool(frame.get("interrupted")),
        pending_work=bool(frame.get("pending_work")),
        error=str(frame.get("error", "")),
        frame=frame,
    )
