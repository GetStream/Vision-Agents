import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.core.harness import Daytona, DefaultHarness
from vision_agents.core.llm.remote import RemoteCall, RemoteEvent, RemotePipelineError
from vision_agents.plugins import stream

SETTLE = 2.0


class Router:
    """A stand-in for the acceleration router, serving the two endpoints a session uses.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive: JSON over HTTP, and frames over a socket.
    """

    def __init__(self):
        self.created: Optional[dict[str, Any]] = None
        self.closed: list[str] = []
        self.commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.url = ""
        self._socket: Optional[web.WebSocketResponse] = None
        self._watching = asyncio.Event()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/agents/sessions", self._create)
        app.router.add_get("/v1/agents/sessions/{id}/events", self._events)
        app.router.add_delete("/v1/agents/sessions/{id}", self._close)
        app.router.add_get("/v1/agents/configs", self._configs)
        return app

    async def _configs(self, _: web.Request) -> web.Response:
        return web.json_response(
            data=[
                {
                    "id": "config-7",
                    "name": "john",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ]
        )

    async def send(self, frame: dict[str, Any]) -> None:
        """Send one frame to whoever is watching the session."""
        await asyncio.wait_for(self._watching.wait(), SETTLE)
        assert self._socket is not None
        await self._socket.send_json(frame)

    async def answered(self) -> dict[str, Any]:
        """The next command the client sent."""
        return await asyncio.wait_for(self.commands.get(), SETTLE)

    async def _create(self, request: web.Request) -> web.Response:
        self.created = await request.json()
        return web.json_response(
            status=201,
            data={
                "id": "session-1",
                "call_id": self.created["call_id"],
                "call_type": self.created.get("call_type", "default"),
                "user_id": self.created.get("user_id", ""),
                "agent_id": self.created.get("agent_id", ""),
                "state": "live",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    async def _close(self, request: web.Request) -> web.Response:
        self.closed.append(request.match_info["id"])
        return web.Response(status=204)

    async def _events(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self._socket = socket
        self._watching.set()

        async for message in socket:
            if message.type == WSMsgType.TEXT:
                await self.commands.put(json.loads(message.data))
        return socket


class TestAccelerated:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    def llm(self, router: Router) -> stream.Accelerated:
        pipeline = stream.Accelerated(
            model="gemma4",
            stt="realtime-best",
            tts="sonic_36",
            url=router.url,
            customer_id="acme",
        )

        @pipeline.register_function(description="The weather where the caller is")
        async def weather(city: str) -> str:
            if city == "nowhere":
                raise ValueError("no such place")
            return f"it is raining in {city}"

        return pipeline

    @pytest.fixture
    def call(self) -> RemoteCall:
        return RemoteCall(
            call_type="default",
            call_id="call-1",
            agent_user_id="agent",
            instructions="be brief",
            harness=DefaultHarness(subagents={"default": "llm-smart"}, vm=Daytona),
            cost_tracking={"project": "moderation", "environment": "dev"},
            memory_filter={"user_id": "222", "company_id": "12312"},
        )

    @pytest.fixture
    async def joined(
        self, llm: stream.Accelerated, call: RemoteCall
    ) -> AsyncIterator[stream.Accelerated]:
        await llm.join_remote(call)
        yield llm
        await llm.leave_remote()

    async def test_the_session_carries_the_models_the_agent_was_given(
        self, router: Router, joined: stream.Accelerated
    ):
        assert router.created is not None
        assert router.created["llm"] == "gemma4"
        assert router.created["stt"] == "realtime-best"
        assert router.created["tts"] == "sonic_36"
        assert router.created["call_id"] == "call-1"
        assert router.created["instructions"] == "be brief"

        async def test_the_session_carries_the_harness_the_agent_was_given(
            self, router: Router, joined: stream.Accelerated
        ):
            assert router.created is not None
            assert router.created["subagent"] == "llm-smart"
            assert router.created["sandbox"] == "daytona"

    async def test_cost_labels_reach_the_session_as_tags(
        self, router: Router, joined: stream.Accelerated
    ):
        assert router.created is not None
        assert router.created["tags"] == {
            "project": "moderation",
            "environment": "dev",
        }

    async def test_the_memory_filter_splits_into_an_identity_and_a_narrowing(
        self, router: Router, joined: stream.Accelerated
    ):
        assert router.created is not None
        assert router.created["memory"]["user_id"] == "222"
        assert router.created["memory"]["filter"] == {"company_id": "12312"}

    async def test_registered_functions_are_offered_to_the_model(
        self, router: Router, joined: stream.Accelerated
    ):
        assert router.created is not None
        tools = router.created["tools"]

        assert [tool["name"] for tool in tools] == ["weather"]
        assert tools[0]["description"] == "The weather where the caller is"
        assert "city" in tools[0]["parameters"]["properties"]

    async def test_what_the_backend_heard_becomes_user_speech(
        self, router: Router, joined: stream.Accelerated
    ):
        events = joined.remote_events()
        await router.send(
            {
                "type": "heard",
                "text": "hello there",
                "participant": {"id": "p1", "user_id": "u1"},
            }
        )

        event = await asyncio.wait_for(anext(events), SETTLE)

        assert event == RemoteEvent(
            type="user_speech", text="hello there", user_id="u1", participant_id="p1"
        )

    async def test_a_finished_turn_reports_whether_it_was_interrupted(
        self, router: Router, joined: stream.Accelerated
    ):
        events = joined.remote_events()
        await router.send({"type": "responded", "text": "hi"})
        await router.send({"type": "turn", "interrupted": True})

        spoken = await asyncio.wait_for(anext(events), SETTLE)
        turn = await asyncio.wait_for(anext(events), SETTLE)

        assert spoken.type == "agent_speech"
        assert spoken.text == "hi"
        assert turn.type == "agent_turn_ended"
        assert turn.interrupted

    async def test_the_call_ending_ends_the_events(
        self, router: Router, joined: stream.Accelerated
    ):
        events = joined.remote_events()
        await router.send({"type": "left", "at": "2026-01-01T00:00:01Z"})

        event = await asyncio.wait_for(anext(events), SETTLE)

        assert event.type == "ended"

    async def test_a_tool_call_is_answered_with_what_the_function_returned(
        self, router: Router, joined: stream.Accelerated
    ):
        await router.send(
            {
                "type": "tool_call",
                "id": "call-9",
                "name": "weather",
                "arguments": json.dumps({"city": "Amsterdam"}),
            }
        )

        answer = await router.answered()

        assert answer == {
            "type": "tool_result",
            "tool_call_id": "call-9",
            "output": "it is raining in Amsterdam",
        }

    async def test_a_tool_that_fails_tells_the_model_so(
        self, router: Router, joined: stream.Accelerated
    ):
        await router.send(
            {
                "type": "tool_call",
                "id": "call-10",
                "name": "weather",
                "arguments": json.dumps({"city": "nowhere"}),
            }
        )

        answer = await router.answered()

        assert answer["tool_call_id"] == "call-10"
        assert "no such place" in answer["error"]

    async def test_speaking_and_answering_go_over_the_session_socket(
        self, router: Router, joined: stream.Accelerated
    ):
        await joined.say_remote("one moment")
        await joined.respond_remote("greet them", interrupt=False)

        assert await router.answered() == {"type": "say", "text": "one moment"}
        assert await router.answered() == {"type": "respond", "text": "greet them"}

    async def test_leaving_closes_the_session(
        self, router: Router, llm: stream.Accelerated, call: RemoteCall
    ):
        await llm.join_remote(call)
        await llm.leave_remote()

        assert await router.answered() == {"type": "close"}

    async def test_a_config_named_when_the_agent_was_built_is_looked_up_on_joining(
        self, router: Router, call: RemoteCall
    ):
        # A config is named where it is defined and identified by id everywhere after, and
        # the lookup waits until joining so an agent can be built before the config exists.
        pipeline = stream.Accelerated(config="john", url=router.url, customer_id="acme")

        await pipeline.join_remote(call)
        await pipeline.leave_remote()

        assert router.created is not None
        assert router.created["config_id"] == "config-7"

    async def test_a_config_name_nothing_is_stored_under_says_so(
        self, router: Router, call: RemoteCall
    ):
        pipeline = stream.Accelerated(
            config="nobody", url=router.url, customer_id="acme"
        )

        with pytest.raises(RemotePipelineError, match="nobody"):
            await pipeline.join_remote(call)
