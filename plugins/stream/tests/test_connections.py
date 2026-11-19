import base64
import json
import secrets
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from vision_agents.plugins import stream

WHEN = "2026-11-19T10:00:00Z"
EXPIRES = "2026-11-19T10:10:00Z"
CONNECTIONS = "/v1/agents/connections"
USER = "X-Stream-User-Id"


class ConnectionsRouter:
    """A stand-in for the router's connection routes, holding connections in memory.

    It applies the router's owner rule for a server-side caller
    (acceleration/internal/api/connections.go, ``ownerOf`` and ``mayReach``): a user's
    connection is filed and reached only while ``X-Stream-User-Id`` names that user, an app's
    by every call. Every request is kept with its method, path, query, body and headers.
    """

    def __init__(self):
        self.url = ""
        self.requests: list[dict[str, Any]] = []
        self.connections: list[dict[str, Any]] = []
        # The most items a page holds, whatever the client asks for, so paging is exercised.
        self.page_size = 2
        # Leaves next_cursor off a page that says there is more, as a broken router might.
        self.drop_cursor = False
        # Puts a next_cursor on the last page too, which a page saying there is no more
        # outranks.
        self.stray_cursor = False

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post(CONNECTIONS, self._create)
        app.router.add_get(CONNECTIONS, self._list)
        app.router.add_get(CONNECTIONS + "/{id}", self._get)
        app.router.add_delete(CONNECTIONS + "/{id}", self._delete)
        app.router.add_post(CONNECTIONS + "/{id}/authorizations", self._authorize)
        return app

    def sent(self, method: str, path: str) -> dict[str, Any]:
        """The latest request to a path."""
        return [
            r for r in self.requests if r["method"] == method and r["path"] == path
        ][-1]

    def add(self, connector: str, user_id: str = "") -> dict[str, Any]:
        """A connection stored as if created earlier, newest first."""
        row = _connection(connector, user_id)
        self.connections.insert(0, row)
        return row

    async def _record(self, request: web.Request) -> dict[str, Any]:
        body = await request.json() if request.can_read_body else None
        asked = {
            "method": request.method,
            "path": request.path,
            "query": dict(request.query),
            "body": body,
            "headers": dict(request.headers),
        }
        self.requests.append(asked)
        return asked

    async def _create(self, request: web.Request) -> web.Response:
        asked = await self._record(request)
        body = asked["body"]
        owner = body["owner"]
        acting = request.headers.get(USER, "")
        if owner["type"] == "app" and "user_id" in owner:
            return _refused(400, "an app-owned connection has no owner.user_id")
        if owner["type"] == "user" and (not acting or owner.get("user_id") != acting):
            return _refused(400, "owner.user_id must be the user this backend acts for")
        row = _connection(body["connector_id"], owner.get("user_id", ""))
        if "label" in body:
            row["label"] = body["label"]
        self.connections.insert(0, row)
        return web.json_response(status=201, data=row)

    async def _list(self, request: web.Request) -> web.Response:
        asked = await self._record(request)
        query = asked["query"]
        acting = request.headers.get(USER, "")
        owned = [
            row
            for row in self.connections
            if row["owner"]["type"] == query.get("owner_type", "app")
            and (row["owner"]["type"] == "app" or row["owner"]["user_id"] == acting)
            and query.get("connector_id", row["connector_id"]) == row["connector_id"]
        ]
        start = int(query.get("cursor", "0"))
        page = owned[start : start + self.page_size]
        more = start + self.page_size < len(owned)
        data: dict[str, Any] = {"items": page, "has_more": more}
        if more and not self.drop_cursor or self.stray_cursor:
            data["next_cursor"] = str(start + self.page_size)
        return web.json_response(data)

    async def _get(self, request: web.Request) -> web.Response:
        await self._record(request)
        row = self._reach(request)
        if row is None:
            return _refused(404, "no such connection", code="connection_not_found")
        return web.json_response(row)

    async def _delete(self, request: web.Request) -> web.Response:
        await self._record(request)
        row = self._reach(request)
        if row is None:
            return _refused(404, "no such connection", code="connection_not_found")
        self.connections.remove(row)
        return web.Response(status=204)

    async def _authorize(self, request: web.Request) -> web.Response:
        await self._record(request)
        row = self._reach(request)
        if row is None:
            return _refused(404, "no such connection", code="connection_not_found")
        return web.json_response(
            status=201,
            data={
                "id": "attempt-1",
                "kind": "consent",
                "launch_url": f"https://router.example/v1/connect/{row['id']}",
                "handoff_token": "handoff-1",
                "expires_at": EXPIRES,
            },
        )

    def _reach(self, request: web.Request) -> Optional[dict[str, Any]]:
        """The connection the path names, if this caller may reach it."""
        acting = request.headers.get(USER, "")
        for row in self.connections:
            if row["id"] != request.match_info["id"]:
                continue
            if row["owner"]["type"] == "app":
                return row
            return row if acting and row["owner"]["user_id"] == acting else None
        return None


def _connection(connector: str, user_id: str) -> dict[str, Any]:
    owner: dict[str, Any] = (
        {"type": "user", "user_id": user_id} if user_id else {"type": "app"}
    )
    return {
        "id": secrets.token_hex(8),
        "connector_id": connector,
        "owner": owner,
        "status": "pending",
        "auth_scheme": "oauth2_code",
        "revision": 1,
        "definition_revision": 1,
        "definition_status": "current",
        "granted_scopes": [],
        "inputs": {},
        "metadata": {},
        "used_by": [],
        "created_at": WHEN,
        "updated_at": WHEN,
    }


def _refused(status: int, message: str, code: str = "invalid_request") -> web.Response:
    return web.json_response(
        status=status,
        data={"error": {"message": message, "type": "invalid_request", "code": code}},
    )


def _claims(authorization: str) -> dict[str, Any]:
    payload = authorization.removeprefix("Bearer ").split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


@pytest.fixture
async def router() -> AsyncIterator[ConnectionsRouter]:
    fake = ConnectionsRouter()
    server = TestServer(fake.app())
    await server.start_server()
    fake.url = str(server.make_url("")).rstrip("/")
    yield fake
    await server.close()


@pytest.fixture
def api(router: ConnectionsRouter) -> stream.Client:
    """A backend holding the app's key and secret, which the router reads the user from."""
    return stream.Client(
        url=router.url, api_key="key", api_secret="secret", authenticate=False
    )


class TestConnections:
    async def test_a_users_connection_names_them_as_owner_and_acting_user(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        created = await api.connections.create("linear", label="work", user_id="alice")

        sent = router.sent("POST", CONNECTIONS)
        assert sent["body"] == {
            "connector_id": "linear",
            "label": "work",
            "owner": {"type": "user", "user_id": "alice"},
        }
        assert sent["headers"][USER] == "alice"
        assert created.owner.user_id == "alice"
        assert created.label == "work"

    async def test_an_app_connection_names_no_user(self, router: ConnectionsRouter):
        # A client built acting for somebody still creates the app's own connection
        # when no user is named on the call.
        api = stream.Client(
            backend=stream.Backend(
                url=router.url,
                api_key="key",
                api_secret="secret",
                authenticate=False,
                user_id="jean",
                acting_for="bob",
            )
        )

        await api.connections.create("linear")

        sent = router.sent("POST", CONNECTIONS)
        assert sent["body"] == {"connector_id": "linear", "owner": {"type": "app"}}
        assert USER not in sent["headers"]

    async def test_every_call_sends_the_user_it_names_and_an_app_call_sends_none(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        mine = await api.connections.create("linear", user_id="alice")
        await api.connections.authorize(mine.id, user_id="alice")
        await api.connections.get(mine.id, user_id="alice")
        await api.connections.list(user_id="alice")
        await api.connections.delete(mine.id, user_id="alice")
        theirs = await api.connections.create("github")
        await api.connections.authorize(theirs.id)
        await api.connections.get(theirs.id)
        await api.connections.list()
        await api.connections.delete(theirs.id)

        sent = [
            (r["method"], r["path"], r["headers"].get(USER)) for r in router.requests
        ]
        assert sent == [
            ("POST", CONNECTIONS, "alice"),
            ("POST", f"{CONNECTIONS}/{mine.id}/authorizations", "alice"),
            ("GET", f"{CONNECTIONS}/{mine.id}", "alice"),
            ("GET", CONNECTIONS, "alice"),
            ("DELETE", f"{CONNECTIONS}/{mine.id}", "alice"),
            ("POST", CONNECTIONS, None),
            ("POST", f"{CONNECTIONS}/{theirs.id}/authorizations", None),
            ("GET", f"{CONNECTIONS}/{theirs.id}", None),
            ("GET", CONNECTIONS, None),
            ("DELETE", f"{CONNECTIONS}/{theirs.id}", None),
        ]

    async def test_the_user_named_on_the_call_replaces_whoever_the_client_acts_for(
        self, router: ConnectionsRouter
    ):
        # Behind the proxy a backend built for a user signs that user's own token; a call
        # naming another user has to stay the app's, naming them in the header.
        api = stream.Client(
            backend=stream.Backend(
                url=router.url,
                api_key="key",
                api_secret="secret",
                authenticate=True,
                user={"id": "jean", "name": "Jean"},
            )
        )

        await api.connections.list(user_id="alice")
        await api.connections.list()

        named, app = router.requests
        assert named["headers"][USER] == "alice"
        assert _claims(named["headers"]["Authorization"])["server"] is True
        assert "user_id" not in _claims(named["headers"]["Authorization"])
        assert USER not in app["headers"]
        assert _claims(app["headers"]["Authorization"])["server"] is True

    async def test_a_call_naming_a_user_reaches_an_app_connection_but_lists_only_theirs(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        apps = router.add("github")
        alices = router.add("linear", user_id="alice")
        router.add("linear", user_id="bob")

        reached = await api.connections.get(apps.get("id"), user_id="alice")
        listed = await api.connections.list(user_id="alice")

        assert reached.id == apps["id"]
        assert [c.id for c in listed] == [alices["id"]]
        assert router.sent("GET", CONNECTIONS)["query"]["owner_type"] == "user"

    async def test_a_users_connection_is_reached_only_as_them(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        alices = router.add("linear", user_id="alice")

        for user in ("", "bob"):
            with pytest.raises(stream.RouterError) as refused:
                await api.connections.get(alices["id"], user_id=user)
            assert refused.value.status == 404
            with pytest.raises(stream.RouterError):
                await api.connections.authorize(alices["id"], user_id=user)
            with pytest.raises(stream.RouterError):
                await api.connections.delete(alices["id"], user_id=user)

        assert router.connections == [alices]

    async def test_authorizing_answers_where_and_how_long_the_consent_runs(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        mine = router.add("linear", user_id="alice")

        consent = await api.connections.authorize(
            mine["id"], return_url="https://app.example/connected", user_id="alice"
        )

        path = f"{CONNECTIONS}/{mine['id']}/authorizations"
        assert router.sent("POST", path)["body"] == {
            "return_url": "https://app.example/connected"
        }
        assert consent.launch_url == f"https://router.example/v1/connect/{mine['id']}"
        assert consent.handoff_token == "handoff-1"
        assert consent.expires_at.isoformat() == "2026-11-19T10:10:00+00:00"

    async def test_authorizing_without_a_return_url_sends_none(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        apps = router.add("github")

        await api.connections.authorize(apps["id"])

        path = f"{CONNECTIONS}/{apps['id']}/authorizations"
        assert router.sent("POST", path)["body"] == {}

    async def test_listing_reads_every_page_of_one_connector(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        router.add("github", user_id="alice")
        wanted = [router.add("linear", user_id="alice") for _ in range(5)]
        router.add("linear")

        listed = await api.connections.list(connector="linear", user_id="alice")

        assert [c.id for c in listed] == [row["id"] for row in reversed(wanted)]
        queries = [r["query"] for r in router.requests]
        assert queries == [
            {"owner_type": "user", "connector_id": "linear", "limit": "200"},
            {
                "owner_type": "user",
                "connector_id": "linear",
                "limit": "200",
                "cursor": "2",
            },
            {
                "owner_type": "user",
                "connector_id": "linear",
                "limit": "200",
                "cursor": "4",
            },
        ]

    async def test_listing_the_apps_asks_for_the_apps_of_every_connector(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        apps = router.add("github")
        router.add("linear", user_id="alice")

        listed = await api.connections.list()

        assert [c.id for c in listed] == [apps["id"]]
        assert router.requests[0]["query"] == {"owner_type": "app", "limit": "200"}

    async def test_a_page_marked_more_without_a_cursor_ends_the_list(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        first = [router.add("github") for _ in range(3)][::-1][:2]
        router.drop_cursor = True

        listed = await api.connections.list()

        assert [c.id for c in listed] == [row["id"] for row in first]
        assert len(router.requests) == 1

    async def test_a_page_marked_last_ends_the_list_whatever_cursor_it_carries(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        apps = router.add("github")
        router.stray_cursor = True

        listed = await api.connections.list()

        assert [c.id for c in listed] == [apps["id"]]
        assert len(router.requests) == 1

    async def test_deleting_a_connection_removes_it(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        mine = router.add("linear", user_id="alice")

        await api.connections.delete(mine["id"], user_id="alice")

        sent = router.sent("DELETE", f"{CONNECTIONS}/{mine['id']}")
        assert sent["query"] == {}
        assert router.connections == []

    @pytest.mark.parametrize("call", ["authorize", "get", "delete"])
    async def test_a_connection_with_no_id_is_refused_before_the_router(
        self, api: stream.Client, router: ConnectionsRouter, call: str
    ):
        methods = {
            "authorize": api.connections.authorize,
            "get": api.connections.get,
            "delete": api.connections.delete,
        }

        with pytest.raises(ValueError, match="needs its id"):
            await methods[call]("", user_id="alice")

        assert router.requests == []

    async def test_a_refusal_is_the_routers_error(
        self, api: stream.Client, router: ConnectionsRouter
    ):
        with pytest.raises(stream.RouterError) as refused:
            await api.connections.get("missing")

        assert refused.value.status == 404
        assert refused.value.code == "connection_not_found"
        assert "reading the connection missing" in str(refused.value)
