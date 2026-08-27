import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.core.telephony import InboundCall
from vision_agents.plugins import stream

SETTLE = 2.0


class Router:
    """A stand-in for the acceleration router, serving the dispatch socket.

    A real server rather than a stub, so what the worker sends is what a router would
    receive and the frames it reads are frames off a socket.
    """

    def __init__(self):
        self.url = ""
        # capacity is what the worker said it could hold, read off the query string.
        self.capacity = ""
        self.reports: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._socket: Optional[web.WebSocketResponse] = None
        self._connected = asyncio.Event()
        self._closing = False

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/dispatch", self._dispatch)
        return app

    async def hand_over(self, frame: dict[str, Any]) -> None:
        """Push one frame to the worker."""
        await asyncio.wait_for(self._connected.wait(), SETTLE)
        assert self._socket is not None
        await self._socket.send_json(frame)

    async def told(self) -> dict[str, Any]:
        """The next thing the worker said."""
        return await asyncio.wait_for(self.reports.get(), SETTLE)

    async def told_of_type(self, kind: str) -> dict[str, Any]:
        """The next thing the worker said of one type, skipping the rest.

        Load is reported on a timer, so a test waiting for an acceptance should not have to
        care whether one landed first.
        """

        async def read() -> dict[str, Any]:
            while True:
                frame = await self.reports.get()
                if frame.get("type") == kind:
                    return frame

        return await asyncio.wait_for(read(), SETTLE)

    async def hang_up(self) -> None:
        """Close the socket, as a router shutting down would."""
        await asyncio.wait_for(self._connected.wait(), SETTLE)
        assert self._socket is not None
        self._closing = True
        await self._socket.close()

    async def _dispatch(self, request: web.Request) -> web.WebSocketResponse:
        self.capacity = request.query.get("capacity", "")
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self._socket = socket
        await socket.send_json({"type": "ready", "worker_id": "worker-7"})
        self._connected.set()

        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            frame = json.loads(message.data)
            await self.reports.put(frame)
            # Answering a ping is what lets the worker measure its own round trip.
            if frame.get("type") == "ping" and not self._closing:
                await socket.send_json({"type": "pong", "at": frame.get("at")})
        return socket


CALL = {
    "type": "call",
    "call_id": "phone-+15125551234",
    "call_type": "default",
    "called_number": "+15125551234",
    "caller_number": "+15550001111",
    "custom": {"line": "support"},
    "at": "2026-08-27T12:00:00Z",
}


class TestStreamDispatch:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    def answered(self) -> asyncio.Queue:
        """The calls a handler was given."""
        return asyncio.Queue()

    @pytest.fixture
    def dispatch(
        self, router: Router, answered: asyncio.Queue
    ) -> stream.StreamDispatch:
        worker = stream.StreamDispatch(
            url=router.url, customer_id="acme", capacity=3, report_every=0.05
        )

        @worker.wait_for_call()
        async def handle(call: InboundCall) -> None:
            await answered.put(call)

        return worker

    @pytest.fixture
    async def waiting(
        self, dispatch: stream.StreamDispatch
    ) -> AsyncIterator[stream.StreamDispatch]:
        """A worker connected and waiting for calls, torn down afterwards."""
        running = asyncio.create_task(dispatch.run())
        yield dispatch
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    async def test_a_worker_says_how_many_calls_it_can_hold(
        self, router: Router, waiting: stream.StreamDispatch
    ):
        # The router passes over a full worker rather than queueing behind it, so this is a
        # promise about the process rather than a hint.
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        assert router.capacity == "3"

    async def test_a_worker_learns_what_the_router_calls_it(
        self, router: Router, waiting: stream.StreamDispatch
    ):
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        async def named() -> None:
            while not waiting.worker_id:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(named(), SETTLE)
        assert waiting.worker_id == "worker-7"

    async def test_an_arriving_call_reaches_the_handler(
        self, router: Router, waiting: stream.StreamDispatch, answered: asyncio.Queue
    ):
        await router.hand_over(CALL)

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.call_id == "phone-+15125551234"
        assert call.call_type == "default"
        assert call.called_number == "+15125551234"
        assert call.caller_number == "+15550001111"
        assert call.custom == {"line": "support"}
        assert call.at is not None
        assert call.at.year == 2026

    async def test_a_call_that_was_handled_is_reported_as_accepted(
        self, router: Router, waiting: stream.StreamDispatch, answered: asyncio.Queue
    ):
        await router.hand_over(CALL)
        await asyncio.wait_for(answered.get(), SETTLE)

        accepted = await router.told_of_type("accepted")

        assert accepted["call_id"] == "phone-+15125551234"

    async def test_a_handler_that_failed_is_reported_as_rejected(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        # A rejection is worth sending because the caller heard a ringing phone that
        # nothing answered, and that is not visible from the router otherwise.
        @dispatch.wait_for_call()
        async def explode(call: InboundCall) -> None:
            raise RuntimeError("no model configured")

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over(CALL)
            rejected = await router.told_of_type("rejected")
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert rejected["call_id"] == "phone-+15125551234"
        assert "no model configured" in rejected["reason"]

    async def test_a_failed_call_does_not_stop_the_next_one(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        seen: list[str] = []

        @dispatch.wait_for_call()
        async def sometimes(call: InboundCall) -> None:
            seen.append(call.call_id)
            if call.call_id == "call-1":
                raise RuntimeError("that one went wrong")

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over({**CALL, "call_id": "call-1"})
            await router.told_of_type("rejected")
            await router.hand_over({**CALL, "call_id": "call-2"})
            await router.told_of_type("accepted")
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert seen == ["call-1", "call-2"]

    async def test_two_calls_are_handled_at_once(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        # Reading the socket is also what delivers the next call, so answering one caller
        # in line would leave the next listening to a ringing phone.
        both = asyncio.Event()
        started = 0

        @dispatch.wait_for_call()
        async def slowly(call: InboundCall) -> None:
            nonlocal started
            started += 1
            if started == 2:
                both.set()
            await both.wait()

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over({**CALL, "call_id": "call-1"})
            await router.hand_over({**CALL, "call_id": "call-2"})
            await asyncio.wait_for(both.wait(), SETTLE)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert started == 2

    async def test_a_worker_reports_what_it_is_doing(
        self, router: Router, waiting: stream.StreamDispatch
    ):
        load = await router.told_of_type("load")

        assert load["active_agents"] == 0
        assert load["cpu_percent"] >= 0.0
        assert load["memory_percent"] >= 0.0
        assert "latency_ms" in load

    async def test_a_report_counts_the_calls_being_handled(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        holding = asyncio.Event()

        @dispatch.wait_for_call()
        async def hold(call: InboundCall) -> None:
            await holding.wait()

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over(CALL)

            async def busy() -> dict[str, Any]:
                while True:
                    load = await router.told_of_type("load")
                    if load["active_agents"] > 0:
                        return load

            load = await asyncio.wait_for(busy(), SETTLE)
        finally:
            holding.set()
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert load["active_agents"] == 1

    async def test_a_worker_times_its_own_round_trip(
        self, router: Router, waiting: stream.StreamDispatch
    ):
        # Measured from this side because this is the side the call's audio has to cross.
        await router.told_of_type("ping")

        async def measured() -> None:
            while waiting._latency_ms <= 0.0:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(measured(), SETTLE)
        assert waiting._latency_ms > 0.0

    async def test_running_without_a_handler_is_refused(self, router: Router):
        # A call would otherwise be taken out of the router's rotation and dropped.
        worker = stream.StreamDispatch(url=router.url, customer_id="acme")

        with pytest.raises(RuntimeError, match="wait_for_call"):
            await worker.run()

    async def test_a_worker_that_can_hold_no_calls_is_refused(self, router: Router):
        with pytest.raises(ValueError, match="cannot answer"):
            stream.StreamDispatch(url=router.url, customer_id="acme", capacity=0)

    async def test_the_router_closing_ends_the_wait(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        running = asyncio.create_task(dispatch.run())
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        await router.hang_up()

        await asyncio.wait_for(running, SETTLE)

    async def test_a_call_still_being_answered_is_waited_for(
        self, router: Router, dispatch: stream.StreamDispatch
    ):
        # Dropping it would hang up on whoever is talking.
        finished = asyncio.Event()
        release = asyncio.Event()

        @dispatch.wait_for_call()
        async def hold(call: InboundCall) -> None:
            await release.wait()
            finished.set()

        running = asyncio.create_task(dispatch.run())
        await router.hand_over(CALL)

        async def answering() -> None:
            while dispatch.active == 0:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(answering(), SETTLE)
        await router.hang_up()
        await asyncio.sleep(0.05)
        assert not running.done(), "the wait should not end while a call is still going"

        release.set()
        await asyncio.wait_for(running, SETTLE)
        assert finished.is_set()

    async def test_a_call_with_no_custom_data_or_time_is_still_answered(
        self, router: Router, waiting: stream.StreamDispatch, answered: asyncio.Queue
    ):
        await router.hand_over({"type": "call", "call_id": "phone-+15125551234"})

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.custom == {}
        assert call.at is None
        assert call.call_type == "default", (
            "a call with no type named is the default one"
        )

    async def test_a_frame_the_worker_does_not_understand_is_ignored(
        self, router: Router, waiting: stream.StreamDispatch, answered: asyncio.Queue
    ):
        await router.hand_over({"type": "something-new"})
        await router.hand_over(CALL)

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.call_id == "phone-+15125551234"
