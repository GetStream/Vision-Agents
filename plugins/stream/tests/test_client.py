import json
from typing import Any, AsyncIterator, Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from vision_agents.plugins import stream

WHEN = "2026-01-01T00:00:00Z"


class Router:
    """A stand-in for the acceleration router: the resource endpoints and the events socket.

    A real server with a real upgrade rather than a stub object, so what is under test is the
    exchange rather than a description of it.
    """

    def __init__(self):
        self.url = ""
        # asked is every request, as "METHOD /path", in order; queries and bodies are kept
        # against the same key so a test can read what it sent.
        self.asked: list[str] = []
        self.queries: dict[str, dict[str, str]] = {}
        self.bodies: dict[str, dict[str, Any]] = {}

        self.configs: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        # pages of response items, each found by the cursor it was handed out under.
        self.pages: list[list[dict[str, Any]]] = []
        # refusals answers a "METHOD /path" with a web.Response built from these keywords
        # instead of its handler, as the router or a proxy in front of it would refuse it.
        self.refusals: dict[str, dict[str, Any]] = {}

    def app(self) -> web.Application:
        app = web.Application(middlewares=[self._refuse])
        app.router.add_get("/v1/agents/configs", self._configs)
        app.router.add_patch("/v1/agents/configs/{id}", self._patch_config)
        app.router.add_post("/v1/agents/sessions", self._create)
        app.router.add_post("/v1/agents/sessions/query", self._query)
        app.router.add_patch("/v1/agents/sessions/{id}", self._update)
        app.router.add_delete("/v1/agents/sessions/{id}", self._no_content)
        app.router.add_post("/v1/agents/sessions/{id}/stop", self._no_content)
        app.router.add_delete("/v1/agents/sessions/{id}/memories", self._no_content)
        app.router.add_delete("/v1/agents/users/{id}/memories", self._no_content)
        app.router.add_post("/v1/agents/sessions/{id}/fork", self._fork)
        app.router.add_post("/v1/agents/sessions/{id}/responses", self._respond)
        app.router.add_get("/v1/agents/sessions/{id}/responses", self._responses)
        app.router.add_get("/v1/agents/sessions/{id}/responses/items", self._items)
        app.router.add_post("/v1/agents/sessions/{id}/rewind", self._rewind)
        app.router.add_post("/v1/agents/guests", self._guest)
        app.router.add_post("/v1/agents/guests/claim", self._claim)
        app.router.add_get("/v1/agents/sessions/{id}/events", self._events)
        app.router.add_post("/v1/agents/simulations", self._simulation)
        app.router.add_get("/v1/agents/simulations", self._simulations)
        app.router.add_get("/v1/agents/simulations/{id}", self._simulation)
        app.router.add_put("/v1/agents/simulations/{id}", self._simulation)
        app.router.add_delete("/v1/agents/simulations/{id}", self._no_content)
        app.router.add_post("/v1/agents/simulations/{id}/run", self._run)
        app.router.add_get("/v1/agents/simulation-runs", self._runs)
        app.router.add_get("/v1/agents/simulation-runs/{id}", self._run)
        app.router.add_post("/v1/agents/simulation-runs/{id}/cancel", self._run)
        return app

    @web.middleware
    async def _refuse(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        refusal = self.refusals.get(f"{request.method} {request.path}")
        if refusal is not None:
            return web.Response(**refusal)
        return await handler(request)

    async def _configs(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(self.configs)

    async def _patch_config(self, request: web.Request) -> web.Response:
        await self._record(request)
        [stored] = [c for c in self.configs if c["id"] == request.match_info["id"]]
        return web.json_response({**stored, **self.bodies[f"PATCH {request.path}"]})

    async def _create(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(status=201, data=self._session("session-1"))

    async def _query(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            {"items": self.sessions, "has_more": True, "next_cursor": "page-2"}
        )

    async def _no_content(self, request: web.Request) -> web.Response:
        await self._record(request)
        if request.match_info.get("id") == "missing":
            return web.json_response(
                status=404,
                data={
                    "error": {
                        "message": "no such thing",
                        "type": "not_found",
                        "code": "not_found",
                        "doc_url": "https://getstream.io/agents/docs/api/errors/#not_found",
                    }
                },
            )
        return web.Response(status=204)

    async def _fork(self, request: web.Request) -> web.Response:
        await self._record(request)
        forked = self._session("session-2")
        forked["forked_from"] = request.match_info["id"]
        return web.json_response(status=201, data=forked)

    async def _update(self, request: web.Request) -> web.Response:
        await self._record(request)
        updated = self._session(request.match_info["id"])
        updated.update(self.bodies[f"PATCH {request.path}"])
        return web.json_response(updated)

    async def _simulation(self, request: web.Request) -> web.Response:
        await self._record(request)
        sent = self.bodies.get(f"{request.method} {request.path}", {})
        simulation = {
            "id": request.match_info.get("id", "simulation-1"),
            "config_id": "config-1",
            "name": "refund",
            "scenario": "Ask for a refund.",
            "assertion": "No refund is promised.",
            "mode": "text",
            "variations": 1,
            "max_turns": 10,
            "created_at": WHEN,
            **sent,
        }
        return web.json_response(
            status=201 if request.method == "POST" else 200, data=simulation
        )

    async def _simulations(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response([])

    async def _run(self, request: web.Request) -> web.Response:
        await self._record(request)
        cancelled = request.path.endswith("/cancel")
        return web.json_response(
            status=202 if request.path.endswith("/run") else 200,
            data=_run(
                "run-1" if request.path.endswith("/run") else request.match_info["id"],
                "cancelled" if cancelled else "running",
            ),
        )

    async def _runs(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response([_run("run-1", "passed")])

    async def _respond(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            status=202,
            data={
                "id": "response-1",
                "session_id": request.match_info["id"],
                "status": "running",
                "said": "Is Stream better?",
                "created_at": WHEN,
            },
        )

    async def _responses(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            {
                "items": [
                    {
                        "id": "response-1",
                        "session_id": request.match_info["id"],
                        "status": "completed",
                        "created_at": WHEN,
                    }
                ],
                "has_more": False,
            }
        )

    async def _items(self, request: web.Request) -> web.Response:
        await self._record(request)
        # The cursor is the index of the page asked for, so a cursor not handed out reads
        # as the wrong page rather than the right one.
        at = int(request.query.get("cursor", "0"))
        page = self.pages[at] if at < len(self.pages) else []
        more = at + 1 < len(self.pages)
        answer: dict[str, Any] = {"items": page, "has_more": more}
        if more:
            answer["next_cursor"] = str(at + 1)
        return web.json_response(answer)

    async def _rewind(self, request: web.Request) -> web.Response:
        await self._record(request)
        # A persistent conversation is what the real router refuses to rewind.
        if request.match_info["id"] == "persistent":
            return web.json_response(
                status=400,
                data={
                    "error": {
                        "message": "a persistent conversation keeps its transcript in Chat",
                        "type": "invalid_request",
                        "code": "invalid_request",
                        "doc_url": "https://getstream.io/agents/docs/api/errors/#invalid_request",
                    }
                },
            )
        return web.Response(status=204)

    async def _guest(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            status=201, data={"id": "guest-1", "token": "guest-token", "name": "Guest"}
        )

    async def _claim(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(
            {"guest_id": "guest-1", "user_id": "jean", "sessions_moved": 3}
        )

    async def _events(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        # Held open until the client goes away, which is what a conversation with nothing
        # happening on it looks like.
        async for _ in socket:
            pass
        return socket

    async def _record(self, request: web.Request) -> None:
        line = f"{request.method} {request.path}"
        self.asked.append(line)
        self.queries[line] = dict(request.query)
        if request.can_read_body:
            self.bodies[line] = await request.json()

    def _session(self, id: str) -> dict[str, Any]:
        return {
            "id": id,
            "agent_id": "agent-1",
            "call_id": "",
            "call_type": "",
            "user_id": "jean",
            "modality": "text",
            "state": "live",
            "conversation_id": "agent:" + id,
            "created_at": WHEN,
        }

    def body(self, method: str, path: str) -> dict[str, Any]:
        return self.bodies[f"{method} {path}"]

    def query(self, method: str, path: str) -> dict[str, str]:
        return self.queries[f"{method} {path}"]

    def requests(self, method: str, path: str) -> int:
        return self.asked.count(f"{method} {path}")


@pytest.fixture
async def router() -> AsyncIterator[Router]:
    fake = Router()
    server = TestServer(fake.app())
    await server.start_server()
    fake.url = str(server.make_url("")).rstrip("/")
    yield fake
    await server.close()


@pytest.fixture
def api(router: Router) -> stream.Client:
    return stream.Client(url=router.url, customer_id="acme")


class TestSessions:
    async def test_a_session_is_opened_against_the_agent_by_name(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create(
            stream.SessionOptions(
                id="0192f5c4-7d1e-7000-8000-000000000001",
                title="Is Stream better?",
                description="The comparison question, again",
                project_id="docs",
                custom={"ticket": "4721"},
                model_overwrites=stream.ModelOverwrites(
                    thinking=stream.ModelOverwritesThinking.HIGH
                ),
            )
        )
        try:
            assert session.id == "session-1"
        finally:
            await session.close()

        body = router.body("POST", "/v1/agents/sessions")
        assert body["id"] == "0192f5c4-7d1e-7000-8000-000000000001"
        assert body["agent"] == "docs"
        assert body["title"] == "Is Stream better?"
        assert body["project_id"] == "docs"
        assert body["custom"] == {"ticket": "4721"}
        assert body["model_overwrites"] == {"thinking": "high"}
        # No call was named, so the conversation is held in writing and kept.
        assert body["text"] is True
        assert body["incognito"] is False

    async def test_an_incognito_session_asks_the_router_to_keep_nothing(
        self, api: stream.Client, router: Router
    ):
        # Every text conversation is kept unless it is incognito, so incognito is the one
        # thing that has to reach the router for nothing to be written down.
        session = await api.agent("docs").sessions.create(
            stream.SessionOptions(incognito=True)
        )
        await session.close()

        body = router.body("POST", "/v1/agents/sessions")
        assert body["incognito"] is True

    async def test_history_the_caller_kept_reaches_the_router_in_order(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create(
            stream.SessionOptions(
                incognito=True,
                history=[
                    stream.HistoryMessage(
                        role=stream.HistoryRole.HISTORY_ROLE_USER,
                        text="Where is order 4471?",
                        name="Ann",
                    ),
                    stream.HistoryMessage(
                        role=stream.HistoryRole.HISTORY_ROLE_ASSISTANT,
                        text="It ships Friday.",
                    ),
                ],
            )
        )
        await session.close()

        body = router.body("POST", "/v1/agents/sessions")
        assert body["history"] == [
            {"role": "user", "text": "Where is order 4471?", "name": "Ann"},
            {"role": "assistant", "text": "It ships Friday."},
        ]

    async def test_connector_bindings_name_a_connection_per_alias(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create(
            stream.SessionOptions(connector_bindings={"crm": "connection-1"})
        )
        await session.close()

        body = router.body("POST", "/v1/agents/sessions")
        assert body["connector_bindings"] == [
            {"name": "crm", "connection_id": "connection-1"}
        ]

    async def test_querying_narrows_to_the_agent_and_the_filters_given(
        self, api: stream.Client, router: Router
    ):
        router.sessions = [router._session("session-1")]

        page = await api.agent("docs").sessions.query(
            stream.Query(
                project_id="docs",
                user_id="jean",
                modality="text",
                state="live",
                agent_id="agent-1",
                limit=50,
                cursor="page-1",
            )
        )

        assert [session.id for session in page.items] == ["session-1"]
        assert page.has_more and page.next_cursor == "page-2"
        assert router.body("POST", "/v1/agents/sessions/query") == {
            "filter": {
                "agent": "docs",
                "project_id": "docs",
                "user_id": "jean",
                "modality": "text",
                "state": "live",
                "agent_id": "agent-1",
            },
            "limit": 50,
            "cursor": "page-1",
        }

    async def test_an_unset_filter_is_left_to_the_router(
        self, api: stream.Client, router: Router
    ):
        await api.agent("docs").sessions.query()

        assert router.body("POST", "/v1/agents/sessions/query") == {
            "filter": {"agent": "docs"}
        }

    async def test_searching_carries_the_phrase_alongside_the_filters(
        self, api: stream.Client, router: Router
    ):
        await api.agent("docs").sessions.search(
            "sendbird comparison", stream.Query(user_id="jean")
        )

        assert router.body("POST", "/v1/agents/sessions/query") == {
            "filter": {
                "agent": "docs",
                "user_id": "jean",
                "text": {"$q": "sendbird comparison"},
            }
        }

    async def test_deleting_a_session_deletes_it(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            await session.delete()
        finally:
            await session.close()

        assert router.requests("DELETE", "/v1/agents/sessions/session-1") == 1

    async def test_a_sessions_memories_are_deleted_by_id(
        self, api: stream.Client, router: Router
    ):
        await api.agent("docs").sessions.delete_memories("session-9")

        assert router.requests("DELETE", "/v1/agents/sessions/session-9/memories") == 1

    async def test_a_delete_the_router_refuses_is_raised(self, api: stream.Client):
        with pytest.raises(stream.RouterError, match="no such thing"):
            await api.agent("docs").sessions.delete("missing")


class TestResponses:
    async def test_asking_something_names_the_turn_it_is_answered_as(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            answer = await session.responses.create("Is Stream better than Sendbird?")
        finally:
            await session.close()

        assert answer.id == "response-1"
        assert answer.status == "running"
        body = router.body("POST", "/v1/agents/sessions/session-1/responses")
        assert body["text"] == "Is Stream better than Sendbird?"

    async def test_a_turns_items_are_narrowed_to_that_turn(
        self, api: stream.Client, router: Router
    ):
        router.pages = [[_item(0)]]
        session = await api.agent("docs").sessions.create()
        try:
            answer = await session.responses.create("Is Stream better?")
            items = await answer.items.all()
        finally:
            await session.close()

        assert [item.ordinal for item in items] == [0]
        query = router.query("GET", "/v1/agents/sessions/session-1/responses/items")
        assert query["response_id"] == "response-1"

    async def test_the_sessions_items_are_every_turns_rather_than_ones(
        self, api: stream.Client, router: Router
    ):
        router.pages = [[_item(0)]]
        session = await api.agent("docs").sessions.create()
        try:
            await session.responses.items.all()
        finally:
            await session.close()

        query = router.query("GET", "/v1/agents/sessions/session-1/responses/items")
        assert "response_id" not in query

    async def test_unwinding_follows_the_cursor_until_there_is_no_more(
        self, api: stream.Client, router: Router
    ):
        # The last page says there is no more, so a fourth request would be asking for a
        # page it has already been told does not exist.
        router.pages = [[_item(0), _item(1)], [_item(2), _item(3)], [_item(4)]]
        session = await api.agent("docs").sessions.create()
        try:
            read = [item async for item in session.responses.items.unwind(limit=2)]
        finally:
            await session.close()

        assert [item.ordinal for item in read] == [0, 1, 2, 3, 4]
        assert (
            router.requests("GET", "/v1/agents/sessions/session-1/responses/items") == 3
        )

    async def test_a_sessions_turns_can_be_read_without_holding_it(
        self, api: stream.Client, router: Router
    ):
        # A conversation that has ended is rows in the backend, so reading it needs no
        # socket and no session handle.
        turns = await api.agent("docs").sessions.responses("session-9").list()

        assert [turn.id for turn in turns.items] == ["response-1"]
        assert not turns.has_more
        assert router.requests("GET", "/v1/agents/sessions/session-9/responses") == 1

    async def test_a_stored_text_conversation_names_every_question(
        self, api: stream.Client, router: Router
    ):
        # The router requires a command id on a user's stored text conversation, and every
        # text conversation is stored unless it is incognito.
        path = "/v1/agents/sessions/session-1/responses"
        session = await api.agent("docs").sessions.create()
        try:
            await session.responses.create("First question")
            first = router.body("POST", path)["command_id"]
            await session.responses.create("Second question")
            second = router.body("POST", path)["command_id"]
            await session.responses.create("Retried question", command_id="request-7")
            retried = router.body("POST", path)["command_id"]
            await session.responses.create(
                "What is this?",
                images=[stream.ImageSource(url="https://example.com/a.png")],
            )
            pictured = router.body("POST", path)
        finally:
            await session.close()

        assert len(first) == 36
        assert first != second, "two questions are two commands"
        assert retried == "request-7"
        assert "command_id" not in pictured, "a command carries text only"

    async def test_a_session_keeping_no_conversation_names_no_command(
        self, api: stream.Client, router: Router
    ):
        await api.agent("docs").sessions.responses("session-9").create("Anyone there?")

        body = router.body("POST", "/v1/agents/sessions/session-9/responses")
        assert "command_id" not in body

    async def test_rewinding_carries_on_from_the_response_given(
        self, api: stream.Client, router: Router
    ):
        # A stored text conversation cannot be rewound, so this is a call's, read back.
        responses = api.agent("docs").sessions.responses("session-3")
        [turn] = (await responses.list()).items

        await responses.rewind(turn)

        body = router.body("POST", "/v1/agents/sessions/session-3/rewind")
        assert body == {"response_id": "response-1"}

    async def test_rewinding_to_an_item_goes_back_to_its_response(
        self, api: stream.Client, router: Router
    ):
        router.pages = [[_item(0)]]
        responses = api.agent("docs").sessions.responses("session-1")
        [item] = await responses.items.all()

        await responses.rewind(item)

        body = router.body("POST", "/v1/agents/sessions/session-1/rewind")
        assert body == {"response_id": "response-1"}

    async def test_a_rewind_the_router_refuses_is_raised(
        self, api: stream.Client, router: Router
    ):
        responses = api.agent("docs").sessions.responses("persistent")

        with pytest.raises(stream.RouterError, match="transcript in Chat"):
            await responses.rewind("response-1")

    async def test_a_response_that_was_never_recorded_cannot_be_rewound_to(
        self, api: stream.Client
    ):
        responses = api.agent("docs").sessions.responses("session-1")

        with pytest.raises(ValueError, match="no id"):
            await responses.rewind("")


class TestUpdate:
    async def test_a_running_session_is_renamed_and_its_models_swapped_at_once(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            updated = await session.update(
                title="Pricing", llm="llm-thinking", thinking="high"
            )
        finally:
            await session.close()

        assert updated.id == "session-1"
        assert updated.title == "Pricing"
        assert updated.llm == "llm-thinking"
        body = router.body("PATCH", "/v1/agents/sessions/session-1")
        assert body == {"title": "Pricing", "llm": "llm-thinking", "thinking": "high"}

    async def test_an_empty_voice_is_sent_since_it_means_the_default(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            await session.update(voice="")
        finally:
            await session.close()

        body = router.body("PATCH", "/v1/agents/sessions/session-1")
        assert body == {"voice": ""}

    async def test_an_ended_session_is_renamed_without_a_handle(
        self, api: stream.Client, router: Router
    ):
        updated = await api.agent("docs").sessions.update(
            "session-9", title="Pricing", custom={"ticket": "4721"}
        )

        assert updated.title == "Pricing"
        body = router.body("PATCH", "/v1/agents/sessions/session-9")
        assert body == {"title": "Pricing", "custom": {"ticket": "4721"}}


class TestMemories:
    async def test_everything_remembered_about_a_user_is_truncated(
        self, api: stream.Client, router: Router
    ):
        await api.memories.truncate("jean")

        assert router.requests("DELETE", "/v1/agents/users/jean/memories") == 1

    async def test_truncating_needs_a_user(self, api: stream.Client, router: Router):
        with pytest.raises(ValueError, match="needs a user id"):
            await api.memories.truncate("")
        assert router.asked == []


class TestSimulations:
    async def test_a_simulation_is_created_run_and_read_back(
        self, api: stream.Client, router: Router
    ):
        simulation = await api.simulations.create(
            stream.SimulationRequest(
                name="refund outside the window",
                config_id="config-1",
                scenario="Ask for a refund 40 days late.",
                assertion="No refund is promised.",
                variations=5,
            )
        )
        run = await api.simulations.run(simulation.id)
        run = await api.simulations.runs.get(run.id)

        assert simulation.name == "refund outside the window"
        assert router.body("POST", "/v1/agents/simulations")["variations"] == 5
        assert run.id == "run-1"
        assert run.state == "running"
        assert router.requests("POST", "/v1/agents/simulations/simulation-1/run") == 1

    async def test_a_simulation_is_replaced_and_deleted_by_id(
        self, api: stream.Client, router: Router
    ):
        updated = await api.simulations.update(
            "simulation-2",
            stream.SimulationRequest(
                name="renamed", config_id="config-1", scenario="s", assertion="a"
            ),
        )
        await api.simulations.delete("simulation-2")

        assert updated.name == "renamed"
        assert router.requests("PUT", "/v1/agents/simulations/simulation-2") == 1
        assert router.requests("DELETE", "/v1/agents/simulations/simulation-2") == 1

    async def test_runs_are_listed_by_simulation_and_cancelled(
        self, api: stream.Client, router: Router
    ):
        [listed] = await api.simulations.runs.list(
            simulation_id="simulation-1", state="passed"
        )
        cancelled = await api.simulations.runs.cancel("run-3")

        assert listed.state == "passed"
        assert router.query("GET", "/v1/agents/simulation-runs") == {
            "simulation_id": "simulation-1",
            "state": "passed",
        }
        assert cancelled.id == "run-3"
        assert cancelled.state == "cancelled"


class TestFork:
    async def test_forking_continues_the_conversation_as_a_new_one(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            forked = await session.fork(
                stream.ForkOptions(
                    title="With a harder model",
                    model_overwrites=stream.ModelOverwrites(llm="openai/gpt-5"),
                )
            )
            await forked.close()
        finally:
            await session.close()

        assert forked.id == "session-2"
        body = router.body("POST", "/v1/agents/sessions/session-1/fork")
        assert body["title"] == "With a harder model"
        assert body["model_overwrites"] == {"llm": "openai/gpt-5"}
        # The history comes across unless the caller says otherwise.
        assert body["messages"] is True

    async def test_forking_without_messages_says_so(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            forked = await session.fork(stream.ForkOptions(messages=False))
            await forked.close()
        finally:
            await session.close()

        assert (
            router.body("POST", "/v1/agents/sessions/session-1/fork")["messages"]
            is False
        )

    async def test_forking_at_a_response_branches_from_there(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            forked = await session.fork(stream.ForkOptions(response_id="response-1"))
            await forked.close()
        finally:
            await session.close()

        body = router.body("POST", "/v1/agents/sessions/session-1/fork")
        assert body["response_id"] == "response-1"
        assert body["messages"] is True

    async def test_a_fork_inherits_the_parents_functions(
        self, api: stream.Client, router: Router
    ):
        agent = api.agent("docs")

        @agent.register()
        async def get_weather(location: str) -> str:
            """Get current weather for a location"""
            return "raining in " + location

        session = await agent.sessions.create()
        try:
            forked = await session.fork()
            await forked.close()
        finally:
            await session.close()

        assert forked.functions.list_functions() == ["get_weather"]

    async def test_the_model_is_offered_the_agents_functions(
        self, api: stream.Client, router: Router
    ):
        agent = api.agent("docs")

        @agent.register()
        async def get_weather(location: str) -> str:
            """Get current weather for a location"""
            return "raining in " + location

        session = await agent.sessions.create()
        await session.close()

        [declared] = router.body("POST", "/v1/agents/sessions")["tools"]
        assert declared["name"] == "get_weather"
        assert "location" in declared["parameters"]["properties"]


class TestChatAndVideo:
    async def test_chat_is_refused_for_a_conversation_that_keeps_no_transcript(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        # What the router said it opened is what the session reports, so this stands in for
        # an incognito conversation: neither has a channel to read.
        session.created.conversation_id = ""
        try:
            with pytest.raises(ValueError, match="no transcript"):
                session.chat()
        finally:
            await session.close()

    async def test_video_is_refused_for_a_conversation_held_in_writing(
        self, api: stream.Client, router: Router
    ):
        session = await api.agent("docs").sessions.create()
        try:
            with pytest.raises(ValueError, match="held in writing"):
                session.video()
        finally:
            await session.close()


class TestAgentConfig:
    async def test_an_agent_is_looked_up_by_the_name_it_is_configured_under(
        self, api: stream.Client, router: Router
    ):
        router.configs = [
            {
                "id": "config-1",
                "name": "docs",
                "mode": "text",
                "created_at": WHEN,
                "updated_at": WHEN,
            }
        ]

        config = await api.agent("docs").config()

        assert config is not None and config.id == "config-1"
        assert router.query("GET", "/v1/agents/configs")["name"] == "docs"

    async def test_a_name_nothing_is_stored_under_is_no_config(
        self, api: stream.Client, router: Router
    ):
        assert await api.agent("nowhere").config() is None

    async def test_updating_the_config_patches_only_what_is_given(
        self, api: stream.Client, router: Router
    ):
        router.configs = [
            {
                "id": "config-1",
                "name": "docs",
                "mode": "text",
                "instructions": "Answer about Stream.",
                "created_at": WHEN,
                "updated_at": WHEN,
            }
        ]

        config = await api.agent("docs").update_config(
            stream.AgentConfigPatch(guardrail="Never discuss pricing.")
        )

        assert config.guardrail == "Never discuss pricing."
        assert config.instructions == "Answer about Stream."
        assert router.body("PATCH", "/v1/agents/configs/config-1") == {
            "guardrail": "Never discuss pricing."
        }

    async def test_updating_an_agent_nothing_is_stored_under_is_refused(
        self, api: stream.Client, router: Router
    ):
        with pytest.raises(stream.RouterError, match="no agent called nowhere"):
            await api.agent("nowhere").update_config(stream.AgentConfigPatch())


class TestGuests:
    async def test_a_guest_is_minted_with_a_token_to_hold(
        self, api: stream.Client, router: Router
    ):
        guest = await api.guest_user(stream.GuestOptions(name="Guest"))

        assert guest.id == "guest-1"
        assert guest.token == "guest-token"
        assert router.body("POST", "/v1/agents/guests")["name"] == "Guest"

    async def test_acting_as_a_guest_is_the_guests_token(self, router: Router):
        # A guest holds a token rather than a customer id, so the backend it acts through
        # has to be one that sends tokens at all.
        api = stream.Client(url=router.url, api_key="key", api_secret="secret")
        guest = stream.GuestUser(id="guest-1", token="guest-token")

        as_guest = api.as_guest(guest)

        assert as_guest.backend.user_id == "guest-1"
        assert as_guest.backend.token == "guest-token"
        assert not as_guest.server_side
        # The client the guest came from is untouched, so a process can hold both.
        assert api.server_side

    async def test_claiming_a_guest_moves_their_conversations(
        self, api: stream.Client, router: Router
    ):
        claimed = await api.claim_guest_user("guest-1", "jean")

        assert claimed.sessions_moved == 3
        body = router.body("POST", "/v1/agents/guests/claim")
        assert body == {"guest_id": "guest-1", "user_id": "jean"}

    async def test_claiming_is_refused_from_a_client_acting_for_one_user(
        self, router: Router
    ):
        api = stream.Client(url=router.url, api_key="key", api_secret="secret")
        as_user = api.as_user("jean", "jean-token")

        with pytest.raises(stream.RouterError, match="server side only"):
            await as_user.claim_guest_user("guest-1", "jean")
        assert router.requests("POST", "/v1/agents/guests/claim") == 0

    async def test_claiming_needs_both_the_guest_and_the_account(
        self, api: stream.Client, router: Router
    ):
        with pytest.raises(stream.RouterError, match="needs the guest"):
            await api.claim_guest_user("guest-1", "")
        assert router.requests("POST", "/v1/agents/guests/claim") == 0


class TestRouterError:
    async def test_a_refusal_carries_the_envelope_and_the_request_id(
        self, api: stream.Client, router: Router
    ):
        router.refusals["POST /v1/agents/sessions/query"] = {
            "status": 400,
            "content_type": "application/json",
            "headers": {"X-Request-Id": "request-1"},
            "text": json.dumps(
                {
                    "error": {
                        "message": "limit must be at most 200",
                        "type": "invalid_request",
                        "code": "validation_failed",
                        "doc_url": "https://getstream.io/agents/docs/api/errors/#validation_failed",
                    }
                }
            ),
        }

        with pytest.raises(stream.RouterError) as raised:
            await api.agent("docs").sessions.query()

        refused = raised.value
        assert str(refused) == "listing the sessions of docs: limit must be at most 200"
        assert refused.status == 400
        assert refused.type == "invalid_request"
        assert refused.code == "validation_failed"
        assert (
            refused.doc_url
            == "https://getstream.io/agents/docs/api/errors/#validation_failed"
        )
        assert refused.request_id == "request-1"

    @pytest.mark.parametrize(
        "status, body, message",
        [
            (502, "<html>bad gateway</html>\n", "<html>bad gateway</html>"),
            (500, '{"error": "the old shape"}', '{"error": "the old shape"}'),
            (503, "", "the router answered 503 Service Unavailable"),
        ],
    )
    async def test_a_body_that_is_not_the_envelope_is_kept_as_the_message(
        self,
        api: stream.Client,
        router: Router,
        status: int,
        body: str,
        message: str,
    ):
        router.refusals["POST /v1/agents/sessions/query"] = {
            "status": status,
            "text": body,
            "headers": {"X-Request-Id": "request-2"},
        }

        with pytest.raises(stream.RouterError) as raised:
            await api.agent("docs").sessions.query()

        refused = raised.value
        assert str(refused) == f"listing the sessions of docs: {message}"
        assert refused.status == status
        assert (refused.type, refused.code, refused.doc_url) == (None, None, None)
        assert refused.request_id == "request-2"

    async def test_a_refused_socket_carries_the_status_and_the_request_id(
        self, api: stream.Client, router: Router
    ):
        router.refusals["GET /v1/agents/sessions/session-1/events"] = {
            "status": 403,
            "content_type": "application/json",
            "headers": {"X-Request-Id": "request-3"},
            "text": json.dumps(
                {
                    "error": {
                        "message": "that session belongs to somebody else",
                        "type": "permission",
                        "code": "forbidden",
                        "doc_url": "https://getstream.io/agents/docs/api/errors/#forbidden",
                    }
                }
            ),
        }

        with pytest.raises(stream.RouterError) as raised:
            await api.agent("docs").sessions.create()

        refused = raised.value
        assert str(refused) == "the router answered 403 Forbidden"
        assert refused.status == 403
        assert refused.request_id == "request-3"
        assert refused.type is None, "aiohttp drops a refused upgrade's body unread"


class TestBackendCredentials:
    def test_a_customer_id_is_the_whole_credential_for_a_router_that_trusts_one(self):
        backend = stream.Backend(url="http://router", customer_id="acme")

        assert backend.headers == {"X-Customer-Id": "acme"}
        assert backend.server_side
        # Stream's own chat and video mean nothing to a router reached this way, so there is
        # no credential to pass on to them.
        assert backend.stream_credentials() is None

    def test_a_key_and_secret_speak_for_the_app_itself(self):
        backend = stream.Backend(
            url="http://router", api_key="key", api_secret="secret"
        )

        headers = backend.headers
        assert headers["X-Api-Key"] == "key"
        assert headers["Stream-Auth-Type"] == "server"
        assert headers["Authorization"].startswith("Bearer ")
        assert backend.server_side

    def test_a_backend_acting_for_a_user_says_which_one(self):
        backend = stream.Backend(
            url="http://router", api_key="key", api_secret="secret", user_id="jean"
        )

        assert backend.headers["X-Stream-User-Id"] == "jean"
        # It still speaks for the app: the header is which of its users it is acting for,
        # not a demotion to being that user.
        assert backend.server_side

    def test_a_token_is_the_whole_credential_and_not_a_backend(self):
        backend = stream.Backend(url="http://router", api_key="key", token="jean-token")

        headers = backend.headers
        assert headers["Authorization"] == "Bearer jean-token"
        assert headers["Stream-Auth-Type"] == "jwt"
        assert not backend.server_side

    def test_the_proxy_gets_the_credential_spelled_streams_way(self):
        backend = stream.Backend(
            url="http://router", api_key="key", api_secret="secret", authenticate=True
        )

        headers = backend.headers
        assert headers["api_key"] == "key"
        # jwt whoever the token is for: the proxy decides who the caller is from the token
        # it verified, so claiming "server" here would be claiming what it is there to
        # decide, and it refuses that.
        assert headers["stream-auth-type"] == "jwt"
        assert "Stream-Auth-Type" not in headers

    def test_acting_for_a_user_gives_chat_and_video_their_token(self):
        backend = stream.Backend(
            url="http://router", api_key="key", api_secret="secret"
        ).as_user({"id": "jean", "name": "Jean"}, "jean-token")

        credentials = backend.stream_credentials()
        assert credentials == {
            "api_key": "key",
            "user": {"id": "jean", "name": "Jean"},
            "token": "jean-token",
        }

    def test_a_backend_with_nothing_to_authenticate_with_is_refused(self, monkeypatch):
        monkeypatch.delenv("STREAM_ACCELERATION_CUSTOMER_ID", raising=False)
        monkeypatch.delenv("STREAM_API_KEY", raising=False)
        monkeypatch.delenv("STREAM_API_SECRET", raising=False)

        with pytest.raises(ValueError, match="who is calling"):
            stream.Backend(url="http://router")

    def test_a_key_without_a_secret_or_a_token_is_refused(self, monkeypatch):
        monkeypatch.delenv("STREAM_API_SECRET", raising=False)

        with pytest.raises(ValueError, match="needs the secret"):
            stream.Backend(url="http://router", api_key="key")

    def test_a_user_needs_an_id_and_a_token(self):
        backend = stream.Backend(
            url="http://router", api_key="key", api_secret="secret"
        )

        with pytest.raises(ValueError, match="needs an id"):
            backend.as_user("", "token")
        with pytest.raises(ValueError, match="no token for"):
            backend.as_user("jean", "")


def _run(id: str, state: str) -> dict[str, Any]:
    return {
        "id": id,
        "simulation_id": "simulation-1",
        "state": state,
        "cases": 1,
        "passed": 0,
        "failed": 0,
        "started_at": WHEN,
    }


def _item(ordinal: int) -> dict[str, Any]:
    return {
        "response_id": "response-1",
        "ordinal": ordinal,
        "kind": "answer",
        "text": f"part {ordinal}",
        "at": WHEN,
    }
