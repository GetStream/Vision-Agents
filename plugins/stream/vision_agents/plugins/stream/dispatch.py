import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Awaitable, Callable, Optional

from vision_agents.core.telephony import InboundCall

from ._backend import Backend
from ._socket import Socket

logger = logging.getLogger(__name__)

DISPATCH_PATH = "/v1/dispatch"

Handler = Callable[[InboundCall], Awaitable[None]]


class StreamDispatch:
    """Waits for inbound calls and runs a handler for each one.

    An inbound call arrives at the router, not here: the caller reached a Stream call over
    SIP and the router found out by webhook. The agent, though, runs in this process. So this
    connects out and waits, and the router pushes a call down the connection when one
    arrives. Nothing has to be publicly reachable for it to work.

    Several workers can wait at once, in which case calls are shared between them.

    Example:
        ```python
        dispatch = StreamDispatch()


        @dispatch.wait_for_call()
        async def answer(call: InboundCall):
            agent = Agent(
                edge=getstream.Edge(),
                agent_user=User(name="John", id="agent"),
                llm=stream.Accelerated(config="john"),
            )
            async with agent.answer(call):
                await agent.simple_response("greet the caller")
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
        self._socket: Optional[Socket] = None
        self._running: set[asyncio.Task[None]] = set()
        # worker_id is what the router calls this connection, for matching a log line here
        # against one there.
        self.worker_id = ""
        # _latency_ms is the last round trip measured to the router. Measured from this side
        # because this is the side the call's audio has to cross.
        self._latency_ms = 0.0
        self._pong = asyncio.Event()

    @property
    def active(self) -> int:
        """How many calls are being handled right now."""
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

    async def run(self) -> None:
        """Wait for calls until cancelled.

        Returns when the router closes the connection. Calls still being handled are waited
        for, because dropping them would hang up on whoever is talking.

        Raises:
            RuntimeError: If no handler has been registered, since a call would then arrive
                with nothing to answer it.
        """
        if self._handler is None:
            raise RuntimeError(
                "register a handler with @dispatch.wait_for_call() before running"
            )

        socket = Socket(
            f"{self.backend.socket(DISPATCH_PATH)}?capacity={self.capacity}",
            self.backend.headers,
        )
        await socket.connect()
        self._socket = socket
        logger.info("waiting for calls on %s", self.backend.url)

        reporter = asyncio.create_task(self._report())
        try:
            await self._read(socket)
        finally:
            reporter.cancel()
            await asyncio.gather(reporter, return_exceptions=True)
            await self._drain()
            await socket.close()
            self._socket = None

    async def _read(self, socket: Socket) -> None:
        """Apply what the router sends until it stops."""
        async for frame in socket.frames():
            if not isinstance(frame, dict):
                continue

            kind = frame.get("type")
            if kind == "call":
                self._answer(_call_of(frame))
            elif kind == "ready":
                self.worker_id = str(frame.get("worker_id", ""))
                logger.info("the router calls this worker %s", self.worker_id)
            elif kind == "pong":
                self._latency_ms = (
                    time.monotonic() - float(frame.get("at", 0.0))
                ) * 1000
                self._pong.set()
            else:
                logger.debug("ignoring a dispatch frame of type %s", kind)

    def _answer(self, call: InboundCall) -> None:
        """Start handling one call.

        The handler runs as a task rather than inline, because reading the socket is also
        what delivers the next call: answering one caller in line would leave the next
        listening to a ringing phone.
        """
        if self._handler is None:
            return

        logger.info(
            "answering a call from %s on %s",
            call.caller_number or "?",
            call.called_number,
        )
        task = asyncio.create_task(self._handle(self._handler, call))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _handle(self, handler: Handler, call: InboundCall) -> None:
        """Run the handler for one call and tell the router how it went.

        Anything the handler raises is caught, because it is somebody else's code and a
        traceback escaping into the task would take the reason with it. The router is told,
        so a call nobody answered shows up there rather than only in this process's log.
        """
        try:
            await handler(call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("a call could not be answered")
            await self._tell(
                {"type": "rejected", "call_id": call.call_id, "reason": str(exc)}
            )
            return
        await self._tell({"type": "accepted", "call_id": call.call_id})

    async def _report(self) -> None:
        """Tell the router how this process is doing, on a timer.

        The router does not use any of it to choose a worker yet. It is sent so that a
        policy which does has numbers to read, and so that an operator can see which worker
        is under load without logging into it.
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
        """Wait for the calls still being handled."""
        running = list(self._running)
        if not running:
            return
        logger.info("waiting for %d call(s) already being answered", len(running))
        await asyncio.gather(*running, return_exceptions=True)


def _call_of(frame: dict[str, object]) -> InboundCall:
    """Read a call frame off the wire."""
    custom = frame.get("custom")
    at = frame.get("at")
    return InboundCall(
        call_id=str(frame.get("call_id", "")),
        call_type=str(frame.get("call_type") or "default"),
        called_number=str(frame.get("called_number", "")),
        caller_number=str(frame.get("caller_number", "")),
        custom={str(key): str(value) for key, value in custom.items()}
        if isinstance(custom, dict)
        else {},
        at=_time_of(at) if isinstance(at, str) else None,
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
