import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Union
from urllib.parse import urlencode

import aiohttp
from vision_agents.core.agents import Agent
from vision_agents.core.llm import FunctionRegistry
from vision_agents.core.messaging import InboundMessage
from vision_agents.core.telephony import InboundCall
from vision_agents.core.utils.utils import await_or_run

from ._backend import Backend
from ._socket import Socket
from .responses import Responses, RouterError
from .sessions import _tools

if TYPE_CHECKING:
    from . import client

logger = logging.getLogger(__name__)

DISPATCH_PATH = "/v1/dispatch"
# FIRST_RETRY and LAST_RETRY bound the wait between attempts to reach a router that dropped
# this worker, in seconds, doubling from one to the other. STEADY_AFTER is how long a
# connection has to have lasted for its loss to start the wait again from FIRST_RETRY.
FIRST_RETRY = 1.0
LAST_RETRY = 30.0
STEADY_AFTER = 60.0

Handler = Callable[[InboundCall], Awaitable[None]]
MessageHandler = Callable[[InboundMessage], Awaitable[None]]
AgentFactory = Callable[[], Union[Agent, Awaitable[Agent]]]


@dataclass
class _Hosting:
    """One set of functions this worker runs for every session under an agent id."""

    agent_id: str
    functions: FunctionRegistry
    tool_timeout: float


class Dispatch:
    """Waits for inbound calls and messages, and runs a handler for each one.

    Neither arrives here first: a caller reached a Stream call over SIP, or somebody wrote in
    a channel, and the router found out by webhook. The agent, though, runs in this process.
    So this connects out and waits, and the router pushes work down the connection when it
    arrives. Nothing has to be publicly reachable for it to work.

    A message arrives here when no agent is running on its channel, or when it was written
    to a running session whose agent leaves text to dispatch. Any other message written to
    an agent that is already running is answered by the router from that session, because
    that agent is the one that knows what has been said so far.

    Several workers can wait at once, in which case the work is shared between them.

    Example:
        ```python
        dispatch = Dispatch()


        @dispatch.wait_for_call()
        async def answer(call: InboundCall):
            agent = Agent(
                edge=getstream.Edge(),
                agent_user=User(name="John", id="agent"),
                llm=stream.Accelerated(config="john"),
            )
            async with agent.answer(call):
                await agent.responses.create("greet the caller")
                await agent.finish()


        asyncio.run(dispatch.run())
        ```
    """

    def __init__(
        self,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
        capacity: int = 4,
        report_every: float = 15.0,
    ):
        """Wait for one customer's calls on a router.

        Args:
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Whose calls to wait for. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
            capacity: How many calls to hold at once. The router passes over a worker that
                is full rather than queueing behind it, so this is a promise about what this
                process can actually answer.
            report_every: How often to tell the router how this process is doing, in
                seconds.

        Raises:
            ValueError: If capacity is not a number of calls.
        """
        if capacity < 1:
            raise ValueError("a worker that can hold no calls cannot answer any")

        self.backend = Backend(url=url, customer_id=customer_id)
        self.capacity = capacity
        self.report_every = report_every

        self._handler: Optional[Handler] = None
        self._message_handler: Optional[MessageHandler] = None
        self._hosted: list[_Hosting] = []
        self._first_retry = FIRST_RETRY
        self._socket: Optional[Socket] = None
        self._running: set[asyncio.Task[None]] = set()
        # The calls and messages alone, which is what the router counts against this
        # worker's capacity. A hosted tool call is not one of them.
        self._handling = 0
        # Which agent is answering which channel. A channel is one conversation, so the
        # agent that answered the last message on it is the one that knows what has been
        # said and should answer the next.
        self._agents: dict[str, Agent] = {}
        self._agents_lock = asyncio.Lock()
        self._started = AsyncExitStack()
        # worker_id is what the router calls this connection, for matching a log line here
        # against one there.
        self.worker_id = ""
        # _latency_ms is the last round trip measured to the router. Measured from this side
        # because this is the side the call's audio has to cross.
        self._latency_ms = 0.0
        self._pong = asyncio.Event()

    @property
    def active(self) -> int:
        """How many calls, messages and hosted tool calls are being handled right now."""
        return len(self._running)

    def wait_for_call(self) -> Callable[[Handler], Handler]:
        """Register what to do with an arriving call.

        The handler is given the call and runs as its own task, so one long call does not
        stop the next from being answered.

        Returns:
            A decorator that keeps the function it is given.
        """

        def register(handler: Handler) -> Handler:
            self._handler = handler
            return handler

        return register

    def wait_for_message(self) -> Callable[[MessageHandler], MessageHandler]:
        """Register what to do with a message written to an agent that is not running, or
        to a running session whose agent leaves text to dispatch.

        The handler is given the message and runs as its own task, the way a call's does.

        Example:
            ```python
            @dispatch.wait_for_message()
            async def written(message: InboundMessage):
                if message.session_id:
                    await dispatch.answer(message)
                    return
                agent = await dispatch.get_or_create_agent(
                    message, lambda: Agent(config="support")
                )
                await agent.responses.create(message.text)
            ```

            Nothing is waited for because the answer is written into the channel by the
            backend as it is generated: the person who wrote is already reading it.

        Returns:
            A decorator that keeps the function it is given.
        """

        def register(handler: MessageHandler) -> MessageHandler:
            self._message_handler = handler
            return handler

        return register

    def host(self, agent: "client.Agent", tool_timeout: float = 0.0) -> None:
        """Run an agent's functions for every session opened under it, whoever opened it.

        A session's own functions run in the process that opened it, which is no use to a
        conversation opened from a browser. Hosting is the other direction: the router offers
        these functions to each session naming the agent and sends every call to a worker
        hosting them. Call before `run`.

        Example:
            ```python
            agent = stream.Client().agent("stream-support")


            @agent.register(description="Read the SDK's source")
            async def investigate_sdk(sdk: str) -> str:
                return await read_source(sdk)


            dispatch.host(agent, tool_timeout=60)
            ```

        Args:
            agent: The agent whose sessions are offered its functions, hosted under its
                name.
            tool_timeout: How long the router waits for one tool call to be answered, in
                seconds, before telling the model it failed. Not how long the worker runs.
                Zero takes the router's default of two minutes.
        """
        self._hosted.append(_Hosting(agent.name, agent.functions, tool_timeout))

    async def get_or_create_agent(
        self, message: InboundMessage, create_agent: AgentFactory
    ) -> Agent:
        """The agent answering on this message's channel, started if none is.

        A channel is one conversation. The second message on it goes to the agent that
        answered the first, which is still open and knows what has been said; only a
        channel nothing is answering calls `create_agent`. The agent is given the channel,
        so what it writes lands in the conversation the question was asked in.

        Agents are kept until this worker stops waiting, so a conversation is not restarted
        between messages.

        Args:
            message: What arrived, whose channel the agent answers in.
            create_agent: Builds the agent for a channel nothing is answering. May be
                sync or async.

        Returns:
            An agent already answering in writing.

        Raises:
            ValueError: If a session is already holding the message's conversation, which
                `answer` is for.
        """
        if message.session_id:
            raise ValueError(
                "a session is already holding this conversation; answer it there with "
                "dispatch.answer(message)"
            )
        async with self._agents_lock:
            answering = self._agents.get(message.channel_id)
            if answering is not None and not answering.closed:
                return answering

            agent = await await_or_run(create_agent)
            await self._started.enter_async_context(agent.chat(message.agent_id))
            self._agents[message.channel_id] = agent
            logger.info("started an agent on %s", message.channel_id)
            return agent

    async def answer(self, message: InboundMessage) -> None:
        """Have the model answer a message written to a running session.

        The response is created with this worker's own credential, acting for whoever wrote
        the message, so it goes to the model rather than back to a worker. It carries the
        message's command, so the answer lands on it.

        Args:
            message: What arrived, naming the session it was written to.

        Raises:
            ValueError: If no session is holding the message.
            RouterError: If the router refuses the response.
        """
        if not message.session_id:
            raise ValueError(
                "no session is holding this message; open one with get_or_create_agent"
            )
        backend = replace(self.backend, acting_for=message.user_id)
        await Responses(backend, message.session_id).create(
            message.text, command_id=message.command_id
        )

    async def run(self) -> None:
        """Wait for calls, messages and hosted tool calls until cancelled.

        Returns when the router closes the connection on purpose. A connection that drops
        any other way, such as a router being redeployed, is opened again and the router is
        told again what this worker hosts. Only the first connection failing is raised.
        Work still being handled is waited for, because dropping a call would hang up on
        whoever is talking.

        Raises:
            RuntimeError: If no handler has been registered and nothing is hosted, since
                work would then arrive with nothing to do it.
            RouterError: If the router refuses the tools this worker hosts.
        """
        if self._handler is None and self._message_handler is None and not self._hosted:
            raise RuntimeError(
                "register a handler with @dispatch.wait_for_call() or "
                "@dispatch.wait_for_message(), or host functions with dispatch.host(), "
                "before running"
            )

        socket = await self._connect()
        logger.info("waiting for work on %s", self.backend.url)

        reporter = asyncio.create_task(self._report())
        try:
            retry = self._first_retry
            while True:
                opened = time.monotonic()
                if not await self._serve(socket):
                    return
                if time.monotonic() - opened >= STEADY_AFTER:
                    retry = self._first_retry

                logger.warning("lost the router, reconnecting in %.1fs", retry)
                while True:
                    await asyncio.sleep(retry)
                    retry = min(retry * 2, LAST_RETRY)
                    try:
                        socket = await self._connect()
                        break
                    except (aiohttp.ClientError, RouterError) as exc:
                        logger.warning(
                            "could not reach the router, retrying in %.1fs: %s",
                            retry,
                            exc,
                        )
        finally:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)
            await self._drain()
            await self._started.aclose()
            self._agents.clear()

    async def _connect(self) -> Socket:
        """Open one dispatch socket, with headers minted for it.

        A token signed when the worker started would have expired by the time a
        long-running one reconnects.
        """
        address = f"{self.backend.socket(DISPATCH_PATH)}?{urlencode(self._waiting())}"
        socket = Socket(address, self.backend.headers)
        try:
            await socket.connect()
        except (aiohttp.ClientError, RouterError):
            await socket.close()
            raise
        return socket

    def _waiting(self) -> dict[str, str]:
        """What this worker says about itself on the way in.

        How much it can hold, how much it is still holding from before a reconnect, and
        which kinds of work it answers. On the handshake because the router may hand work
        over before it has read anything.
        """
        kinds = []
        if self._handler is not None:
            kinds.append("call")
        if self._message_handler is not None:
            kinds.append("message")
        # Always sent, even empty: a worker that only hosts tools answers neither.
        return {
            "capacity": str(self.capacity),
            "active": str(self._handling),
            "handles": ",".join(kinds),
        }

    async def _serve(self, socket: Socket) -> bool:
        """Wait for work on one connection until it ends.

        Returns:
            Whether the connection dropped rather than being closed on purpose, and so is
            worth opening again.
        """
        self._socket = socket
        try:
            await self._read(socket)
            return socket.close_code != aiohttp.WSCloseCode.OK
        finally:
            self._socket = None
            self.worker_id = ""
            await socket.close()

    async def _read(self, socket: Socket) -> None:
        """Apply what the router sends until it stops."""
        async for frame in socket.frames():
            if not isinstance(frame, dict):
                continue

            kind = frame.get("type")
            if kind == "call":
                await self._answer(str(frame.get("work_id", "")), _call_of(frame))
            elif kind == "message":
                await self._reply(str(frame.get("work_id", "")), _message_of(frame))
            elif kind == "ready":
                self.worker_id = str(frame.get("worker_id", ""))
                logger.info("the router calls this worker %s", self.worker_id)
                await self._host()
            elif kind == "tool_call":
                await self._call_hosted(frame)
            elif kind == "hosting":
                logger.info(
                    "the router sends the tools of %s here", frame.get("agent_id", "")
                )
            elif kind == "hosting_refused":
                # Not worth reconnecting: a worker whose tools were refused is one nobody
                # will call, and saying so beats sitting connected looking healthy.
                raise RouterError(
                    f"the router refused to host tools for agent "
                    f"{frame.get('agent_id', '')}: {frame.get('reason', '')}"
                )
            elif kind == "pong":
                self._latency_ms = (
                    time.monotonic() - float(frame.get("at", 0.0))
                ) * 1000
                self._pong.set()
            else:
                logger.debug("ignoring a dispatch frame of type %s", kind)

    async def _host(self) -> None:
        """Tell the router what this worker runs, once it is listening."""
        for offer in self._hosted:
            await self._tell(
                {
                    "type": "host_tools",
                    "agent_id": offer.agent_id,
                    "tools": [tool.to_dict() for tool in _tools(offer.functions)],
                    "timeout_ms": int(offer.tool_timeout * 1000),
                }
            )

    async def _call_hosted(self, frame: dict[str, Any]) -> None:
        """Start running one hosted tool call.

        As a task rather than inline, because the socket it arrived on is also what
        delivers the next one.
        """
        name = str(frame.get("name", ""))
        for offer in self._hosted:
            if offer.functions.get_function(name) is not None:
                break
        else:
            await self._tell(
                {
                    "type": "tool_result",
                    "id": frame.get("id", ""),
                    "error": f"this worker does not run {name}",
                }
            )
            return

        logger.info(
            "running the hosted tool %s for session %s",
            name,
            frame.get("session_id", ""),
        )
        task = asyncio.create_task(self._run_hosted(offer.functions, frame))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _run_hosted(
        self, functions: FunctionRegistry, frame: dict[str, Any]
    ) -> None:
        """Run one hosted tool and answer the router with what it said.

        A failure is reported rather than raised, so the model can say something useful
        about a tool that did not work.
        """
        name = str(frame.get("name", ""))
        result: dict[str, object] = {"type": "tool_result", "id": frame.get("id", "")}
        try:
            arguments = json.loads(frame.get("arguments") or "{}")
            output = await functions.call_function(name, arguments)
            result["output"] = output if isinstance(output, str) else json.dumps(output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("the hosted tool %s failed", name)
            result["error"] = str(exc)
        await self._tell(result)

    async def _answer(self, work_id: str, call: InboundCall) -> None:
        """Start handling one call.

        The handler runs as a task rather than inline, because reading the socket is also
        what delivers the next call: answering one caller in line would leave the next
        listening to a ringing phone.
        """
        if self._handler is None:
            logger.debug("ignoring a call: no handler is registered for one")
            await self._finished(work_id, "this worker answers no calls")
            return

        logger.info(
            "answering a call from %s on %s",
            call.caller_number or "?",
            call.called_number,
        )
        self._handling += 1
        task = asyncio.create_task(self._handle(self._handler, work_id, call))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _reply(self, work_id: str, message: InboundMessage) -> None:
        """Start handling one message, as its own task for the same reason a call is."""
        if self._message_handler is None:
            logger.debug(
                "ignoring a message on %s: no handler is registered for one",
                message.channel_id,
            )
            await self._finished(work_id, "this worker answers no messages")
            return

        logger.info(
            "answering a message from %s on %s",
            message.user_id or "?",
            message.channel_id,
        )
        self._handling += 1
        task = asyncio.create_task(
            self._handle_message(self._message_handler, work_id, message)
        )
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _handle(self, handler: Handler, work_id: str, call: InboundCall) -> None:
        """Run the handler for one call and tell the router how it went.

        Anything the handler raises is caught, because it is somebody else's code and a
        traceback escaping into the task would take the reason with it. The router is told,
        so a call nobody answered shows up there rather than only in this process's log.
        """
        error = ""
        try:
            await handler(call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("a call could not be answered")
            error = str(exc) or type(exc).__name__
        finally:
            self._handling -= 1
        await self._finished(work_id, error)

    async def _handle_message(
        self, handler: MessageHandler, work_id: str, message: InboundMessage
    ) -> None:
        """Run the handler for one message, catching and reporting what it raises."""
        error = ""
        try:
            await handler(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("a message could not be answered")
            error = str(exc) or type(exc).__name__
        finally:
            self._handling -= 1
        await self._finished(work_id, error)

    async def _finished(self, work_id: str, error: str) -> None:
        """Tell the router one piece of work is over, which gives this worker its room back.

        Said even for work this worker had no handler for, because the room it took is held
        until something says it is free.
        """
        done: dict[str, object] = {"type": "done", "work_id": work_id}
        if error:
            done["error"] = error
        await self._tell(done)

    async def _report(self) -> None:
        """Tell the router how this process is doing, on a timer.

        None of it decides where work goes: the router counts what this worker holds from
        what it handed out and what was reported done. This is so an operator can see
        which worker is under load without logging into it.
        """
        while True:
            await asyncio.sleep(self.report_every)
            await self._measure()
            await self._tell(
                {
                    "type": "load",
                    "active_agents": self.active,
                    "cpu_percent": _cpu_percent(),
                    "memory_percent": _memory_percent(),
                    "latency_ms": self._latency_ms,
                }
            )

    async def _measure(self) -> None:
        """Time a round trip to the router.

        The measurement is taken here rather than at the router because this is the side the
        audio has to cross. A pong that does not come back leaves the last figure standing,
        which is more use than a zero.
        """
        self._pong.clear()
        sent = time.monotonic()
        await self._tell({"type": "ping", "at": sent})
        try:
            await asyncio.wait_for(self._pong.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.debug("the router did not answer a ping within 5s")

    async def _tell(self, frame: dict[str, object]) -> None:
        """Send one frame, if the socket is still there.

        A closed socket is not an error here: every one of these is something the router
        would like to know rather than something a call depends on.
        """
        socket = self._socket
        if socket is None or not socket.open:
            return
        try:
            await socket.send(frame)
        except (ConnectionError, RuntimeError) as exc:
            logger.debug("could not reach the router: %s", exc)

    async def _drain(self) -> None:
        """Wait for the work still being handled."""
        running = list(self._running)
        if not running:
            return
        logger.info("waiting for %d already being answered", len(running))
        await asyncio.gather(*running, return_exceptions=True)


def _custom_of(frame: dict[str, object]) -> dict[str, str]:
    """Narrow a frame's custom data to the strings a handler can read.

    The router sends strings, and a value that is not one is dropped rather than rendered,
    because a number that arrived as JSON should not reach a handler as "17.0".
    """
    custom = frame.get("custom")
    if not isinstance(custom, dict):
        return {}
    return {str(key): value for key, value in custom.items() if isinstance(value, str)}


def _call_of(frame: dict[str, object]) -> InboundCall:
    """Read a call frame off the wire."""
    at = frame.get("at")
    return InboundCall(
        call_id=str(frame.get("call_id", "")),
        call_type=str(frame.get("call_type") or "default"),
        called_number=str(frame.get("called_number", "")),
        caller_number=str(frame.get("caller_number", "")),
        custom=_custom_of(frame),
        at=_time_of(at) if isinstance(at, str) else None,
    )


def _message_of(frame: dict[str, object]) -> InboundMessage:
    """Read a message frame off the wire."""
    at = frame.get("at")
    return InboundMessage(
        channel_id=str(frame.get("channel_id", "")),
        channel_type=str(frame.get("channel_type") or "agent"),
        config_id=str(frame.get("config_id", "")),
        custom=_custom_of(frame),
        text=str(frame.get("text", "")),
        message_id=str(frame.get("message_id", "")),
        user_id=str(frame.get("user_id", "")),
        user_name=str(frame.get("user_name", "")),
        at=_time_of(at) if isinstance(at, str) else None,
        session_id=str(frame.get("session_id", "")),
        command_id=str(frame.get("command_id", "")),
    )


def _time_of(text: str) -> Optional[datetime]:
    """Read an RFC 3339 timestamp, tolerating the trailing Z Go writes."""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("a call arrived with an unreadable timestamp: %s", text)
        return None


def _cpu_percent() -> float:
    """How busy the host is, from the standard library rather than a new dependency.

    Load average over the number of cores, so a figure comparable between a laptop and a
    forty-core box. Zero where the platform has no load average, which is honest: an
    invented number would be read as a real one.
    """
    try:
        recent = os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0.0
    cores = os.cpu_count() or 1
    return min(recent / cores * 100.0, 100.0)


def _memory_percent() -> float:
    """How much of the host's memory is in use, in the same spirit as _cpu_percent."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        available = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0.0
    if pages <= 0:
        return 0.0
    return max(0.0, min((pages - available) / pages * 100.0, 100.0))
