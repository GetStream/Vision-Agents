import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
import time

import av
from PIL import Image
from vision_agents.core import Agent, User
from vision_agents.core.harness import Daytona, DefaultHarness
from vision_agents.core.llm.llm import ImageContent
from vision_agents.core.llm.remote import RemoteCall, RemoteEvent, RemotePipelineError
from vision_agents.plugins import stream, getstream

SETTLE = 2.0

# JOHN is the one config the fake has stored, which is what a name is resolved to.
JOHN = {
    "id": "config-7",
    "name": "john",
    "mode": "voice",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


class Router:
    """A stand-in for the acceleration router, serving the two endpoints a session uses.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive: JSON over HTTP, and frames over a socket.
    """

    def __init__(self):
        self.created: Optional[dict[str, Any]] = None
        self.synced: Optional[dict[str, Any]] = None
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
        app.router.add_post("/v1/agents/sync", self._sync)
        return app

    async def _configs(self, _: web.Request) -> web.Response:
        return web.json_response(data=[JOHN])

    async def _sync(self, request: web.Request) -> web.Response:
        self.synced = await request.json()
        return web.json_response({"unchanged": False, "config": JOHN})

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
                "call_id": self.created.get("call_id", ""),
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
            assert router.created["subagents"] == {"default": "llm-smart"}
            assert router.created["sandbox"] == "daytona"

    async def test_named_keyterms_reach_the_session(
        self, router: Router, call: RemoteCall
    ):
        llm = stream.Accelerated(
            model="gemma4",
            url=router.url,
            customer_id="acme",
            keyterms=["Alvarez", "ABC123456"],
        )
        await llm.join_remote(call)
        try:
            assert router.created is not None
            assert router.created["keyterms"] == ["Alvarez", "ABC123456"]
        finally:
            await llm.leave_remote()

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

    async def test_responding_with_images_sends_them_on_the_socket(
        self, router: Router, joined: stream.Accelerated
    ):
        await joined.respond_remote(
            "look",
            interrupt=False,
            images=[ImageContent(data=b"\xff\xd8", detail="low")],
        )

        command = await router.answered()
        assert command["type"] == "respond"
        assert command["text"] == "look"
        assert command["images"][0]["detail"] == "low"
        assert command["images"][0]["url"].startswith("data:image/jpeg;base64,")

    async def test_a_tool_that_returns_an_image_sends_parts(
        self, router: Router, joined: stream.Accelerated
    ):
        @joined.register_function(description="Photograph the shelf")
        async def snapshot() -> dict:
            return {
                "note": "2 roses",
                "photo": ImageContent(data=b"\xff\xd8", mime="image/jpeg"),
            }

        await router.send(
            {
                "type": "tool_call",
                "id": "call-11",
                "name": "snapshot",
                "arguments": "{}",
            }
        )

        answer = await router.answered()
        assert answer["tool_call_id"] == "call-11"
        kinds = [part["type"] for part in answer["output"]]
        assert "text" in kinds
        assert "image_url" in kinds

    async def test_video_selection_is_named_on_the_session(
        self, router: Router, call: RemoteCall
    ):
        pipeline = stream.Accelerated(
            model="gemma4",
            url=router.url,
            customer_id="acme",
            video_source="camera",
            video_max_frames=2,
        )
        await pipeline.join_remote(call)
        try:
            assert router.created is not None
            assert router.created["video"] == {"source": "camera", "max_frames": 2}
        finally:
            await pipeline.leave_remote()

    async def test_capture_is_task_scoped_and_keeps_frame_time(
        self, router: Router, joined: stream.Accelerated
    ):
        agent = Agent(llm=joined, edge=getstream.Edge(), agent_user=User(name="test"))
        at_ms = int(time.time() * 1000)
        agent.observations.append(
            "caller/camera",
            av.VideoFrame.from_image(Image.new("RGB", (2, 2))),
            captured_at_ms=at_ms,
        )
        agent.observations.append(
            "caller/camera",
            av.VideoFrame.from_image(Image.new("RGB", (2, 2), "red")),
            captured_at_ms=at_ms + 10,
        )
        await router.send(
            {
                "type": "tool_call",
                "id": "task-1-capture",
                "name": "get_video_frames",
                "arguments": json.dumps({"at_ms": at_ms, "limit": 1}),
            }
        )
        answer = await router.answered()
        assert answer["tool_call_id"] == "task-1-capture"
        metadata = json.loads(answer["output"][0]["text"])
        assert metadata["frame_id"] == "caller/camera:1"
        assert metadata["captured_at_ms"] == at_ms
        assert answer["output"][1]["type"] == "image_url"

    async def test_slow_work_can_be_cancelled_while_speech_events_continue(
        self, router, joined
    ):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        @joined.register_function(description="Wait for work")
        async def slow() -> str:
            started.set()
            try:
                await never.wait()
                return "late answer"
            finally:
                cancelled.set()

        await router.send(
            {"type": "tool_call", "id": "task-slow", "name": "slow", "arguments": "{}"}
        )
        await asyncio.wait_for(started.wait(), SETTLE)
        await router.send({"type": "heard", "text": "keep talking"})
        events = joined.remote_events()
        event = await asyncio.wait_for(anext(events), SETTLE)
        assert event.text == "keep talking"
        await router.send({"type": "tool_cancel", "id": "task-slow"})
        await asyncio.wait_for(cancelled.wait(), SETTLE)
        assert router.commands.empty()

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

    async def test_what_the_agent_does_not_decide_is_left_for_the_config_to_say(
        self, router: Router
    ):
        # The backend lets the request override the config field by field, so sending an
        # empty instruction or greeting would replace what the config says with nothing.
        pipeline = stream.Accelerated(config="john", url=router.url, customer_id="acme")

        await pipeline.join_remote(
            RemoteCall(
                call_type="default",
                call_id="call-2",
                agent_user_id="agent",
                instructions="",
            )
        )
        await pipeline.leave_remote()

        assert router.created is not None
        assert "instructions" not in router.created
        assert "greeting" not in router.created

    @pytest.fixture
    def john_dir(self, tmp_path, monkeypatch):
        """An agent directory, and the directory an agent naming it is launched from."""
        folder = tmp_path / "agents" / "john"
        folder.mkdir(parents=True)
        (folder / "agent.yaml").write_text("name: john\n")
        (folder / "instructions.md").write_text("Be brief.\n")
        monkeypatch.chdir(tmp_path)
        return folder

    async def test_joining_stores_the_directory_the_config_is_named_after(
        self, router: Router, call: RemoteCall, john_dir
    ):
        # Nobody called sync_agent: the directory is what the config is, so joining is
        # what stores whatever it says now.
        pipeline = stream.Accelerated(config="john", url=router.url, customer_id="acme")

        await pipeline.join_remote(call)
        await pipeline.leave_remote()

        assert router.synced is not None
        assert router.synced["instructions"] == "Be brief."
        assert router.created is not None
        assert router.created["config_id"] == "config-7"

    async def test_joining_again_for_an_untouched_directory_stores_nothing(
        self, router: Router, call: RemoteCall, john_dir
    ):
        first = stream.Accelerated(config="john", url=router.url, customer_id="acme")
        await first.join_remote(call)
        await first.leave_remote()

        router.synced = None
        again = stream.Accelerated(config="john", url=router.url, customer_id="acme")
        await again.join_remote(call)
        await again.leave_remote()

        assert router.synced is None, (
            ".agent_sync should answer for a directory nothing has touched"
        )

    async def test_a_config_name_nothing_is_stored_under_says_so(
        self, router: Router, call: RemoteCall
    ):
        pipeline = stream.Accelerated(
            config="nobody", url=router.url, customer_id="acme"
        )

        with pytest.raises(RemotePipelineError, match="nobody"):
            await pipeline.join_remote(call)

    @pytest.fixture
    async def writing(
        self, llm: stream.Accelerated
    ) -> AsyncIterator[stream.Accelerated]:
        """A pipeline with no call to join, answering in an existing conversation."""
        await llm.join_remote(
            RemoteCall(
                call_type="default",
                call_id="",
                agent_user_id="agent",
                instructions="be brief",
                agent_id="channel-1",
            )
        )
        yield llm
        await llm.leave_remote()

    async def test_a_conversation_with_no_call_to_join_is_held_in_writing(
        self, router: Router, writing: stream.Accelerated
    ):
        assert router.created is not None
        assert router.created["text"] is True
        assert "call_id" not in router.created

    async def test_the_conversation_answered_in_is_the_one_the_agent_was_given(
        self, router: Router, writing: stream.Accelerated
    ):
        # It names the channel the exchange is written into, so answering a message means
        # passing the channel it was written in.
        assert router.created is not None
        assert router.created["agent_id"] == "channel-1"

    async def test_a_reply_arrives_as_it_is_written_and_again_when_it_is_finished(
        self, router: Router, writing: stream.Accelerated
    ):
        events = writing.remote_events()
        await router.send({"type": "response_delta", "text": "Routing picks "})
        await router.send({"type": "responded", "text": "Routing picks a provider."})

        delta = await asyncio.wait_for(anext(events), SETTLE)
        answer = await asyncio.wait_for(anext(events), SETTLE)

        assert delta == RemoteEvent(type="agent_speech_delta", text="Routing picks ")
        assert answer.type == "agent_speech"
        assert answer.text == "Routing picks a provider."

    async def test_work_handed_to_a_skill_is_reported_going_out_and_coming_back(
        self, router: Router, writing: stream.Accelerated
    ):
        events = writing.remote_events()
        await router.send(
            {
                "type": "delegated",
                "task_id": "task-1",
                "skill": "explain",
                "prompt": "failover",
            }
        )
        await router.send(
            {
                "type": "task_settled",
                "task_id": "task-1",
                "skill": "explain",
                "text": "It retries.",
            }
        )

        handed = await asyncio.wait_for(anext(events), SETTLE)
        back = await asyncio.wait_for(anext(events), SETTLE)

        assert handed == RemoteEvent(type="delegated", skill="explain", text="failover")
        assert back == RemoteEvent(
            type="task_settled", skill="explain", text="It retries."
        )

    async def test_what_was_looked_up_is_reported_with_what_it_found(
        self, router: Router, writing: stream.Accelerated
    ):
        events = writing.remote_events()
        await router.send(
            {"type": "looked_up", "query": "delivery cost", "documents": 3}
        )

        event = await asyncio.wait_for(anext(events), SETTLE)

        assert event == RemoteEvent(
            type="looked_up", query="delivery cost", documents=3
        )
