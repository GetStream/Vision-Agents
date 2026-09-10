import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.core import Agent
from vision_agents.core.messaging import InboundMessage
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
        # sessions is one entry per session created, which is how many conversations the
        # worker started rather than carried on.
        self.sessions: list[dict[str, Any]] = []
        self._socket: Optional[web.WebSocketResponse] = None
        self._connected = asyncio.Event()
        self._closing = False

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/dispatch", self._dispatch)
        app.router.add_get("/v1/agents/configs", self._configs)
        app.router.add_post("/v1/agents/sessions", self._create)
        app.router.add_get("/v1/agents/sessions/{id}/events", self._session_events)
        app.router.add_delete("/v1/agents/sessions/{id}", self._close)
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

    async def _configs(self, _: web.Request) -> web.Response:
        return web.json_response(
            data=[
                {
                    "id": "config-7",
                    "name": "chat_desk",
                    "mode": "text",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ]
        )

    async def _create(self, request: web.Request) -> web.Response:
        wanted = await request.json()
        self.sessions.append(wanted)
        return web.json_response(
            status=201,
            data={
                "id": f"session-{len(self.sessions)}",
                "call_id": "",
                "call_type": "agent",
                "user_id": wanted.get("user_id", ""),
                "agent_id": wanted.get("agent_id", ""),
                "text": True,
                "state": "live",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    async def _close(self, _: web.Request) -> web.Response:
        return web.Response(status=204)

    async def _session_events(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for _ in socket:
            pass
        return socket

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

MESSAGE = {
    "type": "message",
    "channel_type": "agent",
    "channel_id": "call-1",
    "agent_id": "call-1",
    "config_id": "chat_desk",
    "text": "is my invoice reissuable?",
    "message_id": "message-1",
    "user_id": "sam",
    "user_name": "Sam",
    "at": "2026-09-08T12:00:00Z",
}


class TestDispatch:
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
    def dispatch(self, router: Router, answered: asyncio.Queue) -> stream.Dispatch:
        worker = stream.Dispatch(
            url=router.url, customer_id="acme", capacity=3, report_every=0.05
        )

        @worker.wait_for_call()
        async def handle(call: InboundCall) -> None:
            await answered.put(call)

        return worker

    @pytest.fixture
    async def waiting(
        self, dispatch: stream.Dispatch
    ) -> AsyncIterator[stream.Dispatch]:
        """A worker connected and waiting for calls, torn down afterwards."""
        running = asyncio.create_task(dispatch.run())
        yield dispatch
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    async def test_a_worker_says_how_many_calls_it_can_hold(
        self, router: Router, waiting: stream.Dispatch
    ):
        # The router passes over a full worker rather than queueing behind it, so this is a
        # promise about the process rather than a hint.
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        assert router.capacity == "3"

    async def test_a_worker_learns_what_the_router_calls_it(
        self, router: Router, waiting: stream.Dispatch
    ):
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        async def named() -> None:
            while not waiting.worker_id:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(named(), SETTLE)
        assert waiting.worker_id == "worker-7"

    async def test_an_arriving_call_reaches_the_handler(
        self, router: Router, waiting: stream.Dispatch, answered: asyncio.Queue
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
        self, router: Router, waiting: stream.Dispatch, answered: asyncio.Queue
    ):
        await router.hand_over(CALL)
        await asyncio.wait_for(answered.get(), SETTLE)

        accepted = await router.told_of_type("accepted")

        assert accepted["call_id"] == "phone-+15125551234"

    async def test_a_handler_that_failed_is_reported_as_rejected(
        self, router: Router, dispatch: stream.Dispatch
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
        self, router: Router, dispatch: stream.Dispatch
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
        self, router: Router, dispatch: stream.Dispatch
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
        self, router: Router, waiting: stream.Dispatch
    ):
        load = await router.told_of_type("load")

        assert load["active_agents"] == 0
        assert load["cpu_percent"] >= 0.0
        assert load["memory_percent"] >= 0.0
        assert "latency_ms" in load

    async def test_a_report_counts_the_calls_being_handled(
        self, router: Router, dispatch: stream.Dispatch
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
        self, router: Router, waiting: stream.Dispatch
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
        worker = stream.Dispatch(url=router.url, customer_id="acme")

        with pytest.raises(RuntimeError, match="wait_for_call"):
            await worker.run()

    async def test_the_refusal_says_a_message_handler_would_also_do(
        self, router: Router
    ):
        worker = stream.Dispatch(url=router.url, customer_id="acme")

        with pytest.raises(RuntimeError, match="wait_for_message"):
            await worker.run()

    async def test_a_worker_that_can_hold_no_calls_is_refused(self, router: Router):
        with pytest.raises(ValueError, match="cannot answer"):
            stream.Dispatch(url=router.url, customer_id="acme", capacity=0)

    async def test_the_router_closing_ends_the_wait(
        self, router: Router, dispatch: stream.Dispatch
    ):
        running = asyncio.create_task(dispatch.run())
        await asyncio.wait_for(router._connected.wait(), SETTLE)

        await router.hang_up()

        await asyncio.wait_for(running, SETTLE)

    async def test_a_call_still_being_answered_is_waited_for(
        self, router: Router, dispatch: stream.Dispatch
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
        self, router: Router, waiting: stream.Dispatch, answered: asyncio.Queue
    ):
        await router.hand_over({"type": "call", "call_id": "phone-+15125551234"})

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.custom == {}
        assert call.at is None
        assert call.call_type == "default", (
            "a call with no type named is the default one"
        )

    async def test_a_frame_the_worker_does_not_understand_is_ignored(
        self, router: Router, waiting: stream.Dispatch, answered: asyncio.Queue
    ):
        await router.hand_over({"type": "something-new"})
        await router.hand_over(CALL)

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.call_id == "phone-+15125551234"

    async def test_an_arriving_message_reaches_the_message_handler(
        self, router: Router, dispatch: stream.Dispatch
    ):
        written: asyncio.Queue = asyncio.Queue()

        @dispatch.wait_for_message()
        async def read(message: InboundMessage) -> None:
            await written.put(message)

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over(MESSAGE)
            message = await asyncio.wait_for(written.get(), SETTLE)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert message.channel_id == "call-1"
        assert message.channel_type == "agent"
        assert message.config_id == "chat_desk"
        assert message.text == "is my invoice reissuable?"
        assert message.message_id == "message-1"
        assert message.user_id == "sam"
        assert message.user_name == "Sam"
        assert message.at is not None
        assert message.at.year == 2026

    async def test_the_channel_a_message_arrived_on_is_the_agent_to_answer_as(
        self, router: Router, dispatch: stream.Dispatch
    ):
        # Passing this to a new session is what puts the answer back in the conversation
        # the question was asked in.
        written: asyncio.Queue = asyncio.Queue()

        @dispatch.wait_for_message()
        async def read(message: InboundMessage) -> None:
            await written.put(message)

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over(MESSAGE)
            message = await asyncio.wait_for(written.get(), SETTLE)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert message.agent_id == "call-1"

    async def test_a_message_is_ignored_by_a_worker_that_only_answers_calls(
        self, router: Router, waiting: stream.Dispatch, answered: asyncio.Queue
    ):
        # Nothing is reported back: there is no line anybody is waiting on, unlike a call.
        await router.hand_over(MESSAGE)
        await router.hand_over(CALL)

        call = await asyncio.wait_for(answered.get(), SETTLE)

        assert call.call_id == "phone-+15125551234"

    async def test_a_message_handler_that_failed_does_not_stop_the_next_one(
        self, router: Router, dispatch: stream.Dispatch
    ):
        seen: list[str] = []
        both = asyncio.Event()

        @dispatch.wait_for_message()
        async def sometimes(message: InboundMessage) -> None:
            seen.append(message.text)
            if len(seen) == 2:
                both.set()
            if message.text == "first":
                raise RuntimeError("that one went wrong")

        running = asyncio.create_task(dispatch.run())
        try:
            await router.hand_over({**MESSAGE, "text": "first"})
            await router.hand_over({**MESSAGE, "text": "second"})
            await asyncio.wait_for(both.wait(), SETTLE)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert seen == ["first", "second"]

    async def test_a_worker_can_wait_for_messages_without_answering_calls(
        self, router: Router
    ):
        # An agent that only answers in writing has no reason to be handed a phone call.
        worker = stream.Dispatch(url=router.url, customer_id="acme", report_every=0.05)
        written: asyncio.Queue = asyncio.Queue()

        @worker.wait_for_message()
        async def read(message: InboundMessage) -> None:
            await written.put(message)

        running = asyncio.create_task(worker.run())
        try:
            await router.hand_over(MESSAGE)
            message = await asyncio.wait_for(written.get(), SETTLE)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

        assert message.text == "is my invoice reissuable?"

    @pytest.fixture
    def acceleration(self, router: Router, monkeypatch: pytest.MonkeyPatch) -> None:
        """The environment an agent built from a stored config reads."""
        monkeypatch.setenv("STREAM_ACCELERATION_URL", router.url)
        monkeypatch.setenv("STREAM_ACCELERATION_CUSTOMER_ID", "acme")
        monkeypatch.setenv("STREAM_API_KEY", "key")
        monkeypatch.setenv("STREAM_API_SECRET", "secret")

    async def answering(
        self, router: Router, dispatch: stream.Dispatch, *channels: str
    ) -> list[Agent]:
        """The agent each of those channels was answered by, in order."""
        answered: asyncio.Queue[Agent] = asyncio.Queue()

        @dispatch.wait_for_message()
        async def read(message: InboundMessage) -> None:
            await answered.put(
                await dispatch.get_or_create_agent(
                    message, lambda: Agent(config="chat_desk")
                )
            )

        running = asyncio.create_task(dispatch.run())
        try:
            agents = []
            for channel in channels:
                await router.hand_over({**MESSAGE, "channel_id": channel})
                agents.append(await asyncio.wait_for(answered.get(), SETTLE))
            return agents
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    async def test_an_agent_answers_in_the_channel_the_message_was_written_in(
        self, router: Router, dispatch: stream.Dispatch, acceleration: None
    ):
        # The channel is the agent id, which is what puts the answer back in the
        # conversation the question was asked in.
        await self.answering(router, dispatch, "call-1")

        assert len(router.sessions) == 1
        assert router.sessions[0]["agent_id"] == "call-1"
        assert router.sessions[0]["text"] is True
        assert router.sessions[0]["config_id"] == "config-7"

    async def test_a_second_message_on_a_channel_goes_to_the_agent_that_answered_the_first(
        self, router: Router, dispatch: stream.Dispatch, acceleration: None
    ):
        # Starting a second agent would answer as though the first exchange never happened.
        first, second = await self.answering(router, dispatch, "call-1", "call-1")

        assert first is second
        assert len(router.sessions) == 1

    async def test_another_channel_is_another_conversation(
        self, router: Router, dispatch: stream.Dispatch, acceleration: None
    ):
        first, second = await self.answering(router, dispatch, "call-1", "call-2")

        assert first is not second
        assert [created["agent_id"] for created in router.sessions] == [
            "call-1",
            "call-2",
        ]

    async def test_a_message_still_being_answered_is_waited_for(
        self, router: Router, dispatch: stream.Dispatch
    ):
        finished = asyncio.Event()
        release = asyncio.Event()

        @dispatch.wait_for_message()
        async def hold(message: InboundMessage) -> None:
            await release.wait()
            finished.set()

        running = asyncio.create_task(dispatch.run())
        await router.hand_over(MESSAGE)

        async def answering() -> None:
            while dispatch.active == 0:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(answering(), SETTLE)
        await router.hang_up()
        await asyncio.sleep(0.05)
        assert not running.done(), "the wait should not end mid-answer"

        release.set()
        await asyncio.wait_for(running, SETTLE)
        assert finished.is_set()
