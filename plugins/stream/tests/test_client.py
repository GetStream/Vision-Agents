import json
from typing import Any, AsyncIterator

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
        # pages of response items, handed out one request at a time.
        self.pages: list[list[dict[str, Any]]] = []
        self.given = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/agents/configs", self._configs)
        app.router.add_post("/v1/agents/sessions", self._create)
        app.router.add_get("/v1/agents/sessions", self._list)
        app.router.add_get("/v1/agents/sessions/search", self._list)
        app.router.add_delete("/v1/agents/sessions/{id}", self._close)
        app.router.add_post("/v1/agents/sessions/{id}/fork", self._fork)
        app.router.add_post("/v1/agents/sessions/{id}/responses", self._respond)
        app.router.add_get("/v1/agents/sessions/{id}/responses", self._responses)
        app.router.add_get("/v1/agents/sessions/{id}/responses/items", self._items)
        app.router.add_post("/v1/agents/guests", self._guest)
        app.router.add_post("/v1/agents/guests/claim", self._claim)
        app.router.add_get("/v1/agents/sessions/{id}/events", self._events)
        return app

    async def _configs(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(self.configs)

    async def _create(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(status=201, data=self._session("session-1"))

    async def _list(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(self.sessions)

    async def _close(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response({})

    async def _fork(self, request: web.Request) -> web.Response:
        await self._record(request)
        forked = self._session("session-2")
        forked["forked_from"] = request.match_info["id"]
        return web.json_response(status=201, data=forked)

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
            [
                {
                    "id": "response-1",
                    "session_id": request.match_info["id"],
                    "status": "completed",
                    "created_at": WHEN,
                }
            ]
        )

    async def _items(self, request: web.Request) -> web.Response:
        await self._record(request)
        if self.given >= len(self.pages):
            return web.json_response([])
        page = self.pages[self.given]
        self.given += 1
        return web.json_response(page)

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
                title="Is Stream better?",
                description="The comparison question, again",
                project="docs",
                custom={"ticket": "4721"},
                persist=True,
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
        assert body["agent"] == "docs"
        assert body["title"] == "Is Stream better?"
        assert body["project"] == "docs"
        assert body["custom"] == {"ticket": "4721"}
        assert body["model_overwrites"] == {"thinking": "high"}
        # No call was named, so the conversation is held in writing and kept.
        assert body["text"] is True
        assert body["persist_conversation"] is True

    async def test_an_incognito_session_never_asks_for_a_transcript(
        self, api: stream.Client, router: Router
    ):
        # Asking for both is a contradiction, and the conversation the caller wanted is the
        # incognito one: an off-the-record conversation writes nothing down by definition.
        session = await api.agent("docs").sessions.create(
            stream.SessionOptions(incognito=True, persist=True)
        )
        await session.close()

        body = router.body("POST", "/v1/agents/sessions")
        assert body["incognito"] is True
        assert "persist_conversation" not in body

    async def test_querying_narrows_to_the_agent_and_the_filters_given(
        self, api: stream.Client, router: Router
    ):
        router.sessions = [router._session("session-1")]

        listed = await api.agent("docs").sessions.query(
            stream.Query(
                project="docs",
                user_id="jean",
                state="closed",
                custom={"ticket": "4721"},
                limit=50,
            )
        )

        assert [session.id for session in listed] == ["session-1"]
        query = router.query("GET", "/v1/agents/sessions")
        assert query["agent"] == "docs"
        assert query["project"] == "docs"
        assert query["user_id"] == "jean"
        assert query["state"] == "closed"
        assert json.loads(query["custom"]) == {"ticket": "4721"}
        assert query["limit"] == "50"
        # An unset offset leaves the router's own default rather than a zero this end
        # invented.
        assert "offset" not in query

    async def test_searching_carries_the_phrase_alongside_the_filters(
        self, api: stream.Client, router: Router
    ):
        router.sessions = [router._session("session-1")]

        await api.agent("docs").sessions.search(
            "sendbird comparison", stream.Query(project="docs")
        )

        query = router.query("GET", "/v1/agents/sessions/search")
        assert query["q"] == "sendbird comparison"
        assert query["agent"] == "docs"
        assert query["project"] == "docs"


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

    async def test_unwinding_pages_until_a_short_page_arrives(
        self, api: stream.Client, router: Router
    ):
        # Two full pages and a short one. The short page is the end, so a fourth request
        # would be asking for a page it has already been told does not exist.
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

        assert [turn.id for turn in turns] == ["response-1"]
        assert router.requests("GET", "/v1/agents/sessions/session-9/responses") == 1


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


def _item(ordinal: int) -> dict[str, Any]:
    return {
        "response_id": "response-1",
        "ordinal": ordinal,
        "kind": "answer",
        "text": f"part {ordinal}",
        "at": WHEN,
    }
