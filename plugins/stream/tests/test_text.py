import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.plugins import stream

SETTLE = 2.0


class Router:
    """A stand-in for the acceleration router, serving what a text session uses.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive: JSON over HTTP, and frames over a socket.
    """

    def __init__(self):
        self.created: Optional[dict[str, Any]] = None
        self.commands: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.url = ""
        self._socket: Optional[web.WebSocketResponse] = None
        self._watching = asyncio.Event()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/agents/sessions", self._create)
        app.router.add_get("/v1/agents/sessions/{id}/events", self._events)
        app.router.add_delete("/v1/agents/sessions/{id}", self._close)
        return app

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
                "call_id": "",
                "call_type": "default",
                "user_id": "vision-agent",
                "agent_id": "agent-1",
                "text": True,
                "state": "live",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    async def _close(self, request: web.Request) -> web.Response:
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


class TestTextSession:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    async def session(self, router: Router) -> AsyncIterator[stream.TextSession]:
        held = stream.TextSession(
            config_id="docs-agent",
            skills=["explain"],
            url=router.url,
            customer_id="acme",
        )
        await held.start()
        yield held
        await held.close()

    async def asking(
        self, session: stream.TextSession, question: str
    ) -> tuple[asyncio.Task, list[stream.TextEvent]]:
        """Start a question and collect what comes back as it arrives."""
        seen: list[stream.TextEvent] = []

        async def read() -> None:
            async for event in session.ask(question):
                seen.append(event)

        return asyncio.create_task(read()), seen

    async def test_a_conversation_in_writing_joins_no_call(
        self, session: stream.TextSession, router: Router
    ):
        assert router.created is not None
        assert router.created["text"] is True
        assert "call_id" not in router.created
        assert router.created["config_id"] == "docs-agent"
        assert router.created["skill_names"] == ["explain"]

    async def test_an_answer_arrives_as_it_is_written(
        self, session: stream.TextSession, router: Router
    ):
        reader, seen = await self.asking(session, "what is routing")

        asked = await router.answered()
        assert asked == {"type": "respond", "text": "what is routing"}

        await router.send({"type": "response_delta", "text": "Routing picks "})
        await router.send({"type": "response_delta", "text": "a provider."})
        await router.send({"type": "responded", "text": "Routing picks a provider."})
        await asyncio.wait_for(reader, SETTLE)

        assert [event.type for event in seen] == ["delta", "delta", "answer"]
        assert "".join(event.text for event in seen if event.type == "delta") == (
            "Routing picks a provider."
        )
        assert seen[-1].text == "Routing picks a provider."

    async def test_a_question_handed_to_a_skill_is_answered_over_two_turns(
        self, session: stream.TextSession, router: Router
    ):
        # The model says something while the work runs, and the answer arrives once it
        # comes back. Ending at the first reply would hand back the filler and drop the
        # answer the reader asked for.
        reader, seen = await self.asking(session, "how does failover work")
        await router.answered()

        await router.send(
            {
                "type": "delegated",
                "task_id": "task-1",
                "skill": "explain",
                "prompt": "failover",
            }
        )
        await router.send({"type": "responded", "text": "Let me check that."})
        await router.send(
            {
                "type": "task_settled",
                "task_id": "task-1",
                "skill": "explain",
                "text": "It retries.",
            }
        )
        await router.send(
            {"type": "responded", "text": "It retries on the next provider."}
        )
        await asyncio.wait_for(reader, SETTLE)

        assert [event.type for event in seen] == [
            "delegated",
            "answer",
            "settled",
            "answer",
        ]
        assert seen[0].skill == "explain"
        assert seen[-1].text == "It retries on the next provider."

    async def test_what_was_looked_up_is_reported_with_what_it_found(
        self, session: stream.TextSession, router: Router
    ):
        reader, seen = await self.asking(session, "what does delivery cost")
        await router.answered()

        await router.send(
            {"type": "looked_up", "query": "delivery cost", "documents": 3}
        )
        await router.send({"type": "responded", "text": "Free over $50."})
        await asyncio.wait_for(reader, SETTLE)

        assert seen[0].type == "looked_up"
        assert seen[0].query == "delivery cost"
        assert seen[0].documents == 3

    async def test_the_conversation_ends_when_the_backend_leaves(
        self, session: stream.TextSession, router: Router
    ):
        # A session closed underneath the reader must not leave it waiting for an answer
        # that is never coming.
        reader, seen = await self.asking(session, "anything")
        await router.answered()

        await router.send({"type": "left"})
        await asyncio.wait_for(reader, SETTLE)

        assert seen == []

    async def test_closing_tells_the_backend_the_conversation_is_over(
        self, router: Router
    ):
        held = stream.TextSession(url=router.url, customer_id="acme")
        await held.start()

        await held.close()

        assert await router.answered() == {"type": "close"}
        assert held.session is None
