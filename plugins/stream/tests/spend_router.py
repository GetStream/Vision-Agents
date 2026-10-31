"""A stand-in for the router's spend routes: limits, grants, usage, teams, the account and spend.

A real aiohttp server keeping limits and grants in memory the way the router answers them, so
what is under test is the exchange rather than a description of it.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import MultiDictProxy

WHEN = "2026-11-01T00:00:00Z"
WINDOW_END = "2026-11-02T00:00:00Z"

# The ids a limit's subject is named by, in the order the router lists them.
IDS = ("app_id", "api_key", "user_id", "team", "category")
APP_SUBJECTS = ("app", "app_category", "app_team")


def refused(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        status=status,
        data={
            "error": {
                "message": message,
                "type": "invalid_request",
                "code": code,
                "doc_url": f"https://getstream.io/agents/docs/api/errors/#{code}",
            }
        },
    )


class SpendRouter:
    def __init__(self):
        self.url = ""
        # asked is every request, as "METHOD /path", in order; queries and bodies are kept
        # against the same key, the latest of each, so a test can read what it sent.
        self.asked: list[str] = []
        self.queries: dict[str, MultiDictProxy[str]] = {}
        self.bodies: dict[str, dict[str, Any]] = {}
        self.headers: dict[str, dict[str, str]] = {}

        self.limits: list[dict[str, Any]] = []
        self.grants: list[dict[str, Any]] = []
        self.teams: list[dict[str, Any]] = []
        self.account: dict[str, Any] = account()
        self.credits: list[dict[str, Any]] = []
        self.buckets: list[dict[str, Any]] = []
        # The most items a page holds, whatever the client asks for, so paging is exercised.
        self.page_size = 200
        # How many limit lists still answer none, as a list read before another writer's
        # create landed would.
        self.lists_before_create = 0
        self._ids = 0

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/spend/limits", self._list_limits)
        app.router.add_post("/v1/spend/limits", self._create_limit)
        app.router.add_get("/v1/spend/limits/grants", self._list_grants)
        app.router.add_post("/v1/spend/limits/grants", self._create_grant)
        app.router.add_post("/v1/spend/limits/grants/{id}/revoke", self._revoke)
        app.router.add_get("/v1/spend/limits/usage", self._usage)
        app.router.add_post("/v1/spend/limits/usage/reset", self._reset)
        app.router.add_patch("/v1/spend/limits/{id}", self._update_limit)
        app.router.add_delete("/v1/spend/limits/{id}", self._delete_limit)
        app.router.add_get("/v1/teams", self._teams)
        app.router.add_get("/v1/account", self._account)
        app.router.add_get("/v1/account/credits", self._credits)
        app.router.add_get("/v1/stats/spend", self._spend)
        return app

    def body(self, method: str, path: str) -> dict[str, Any]:
        return self.bodies[f"{method} {path}"]

    def query(self, method: str, path: str) -> MultiDictProxy[str]:
        return self.queries[f"{method} {path}"]

    def requests(self, method: str, path: str) -> int:
        return self.asked.count(f"{method} {path}")

    def add_limit(self, **fields: Any) -> dict[str, Any]:
        """A limit the router already keeps."""
        row = limit(self._next(), **fields)
        self.limits.append(row)
        return row

    def add_grant(self, **fields: Any) -> dict[str, Any]:
        row = grant(self._next(), **fields)
        self.grants.append(row)
        return row

    async def _record(self, request: web.Request) -> None:
        line = f"{request.method} {request.path}"
        self.asked.append(line)
        self.queries[line] = request.query
        self.headers[line] = dict(request.headers)
        if request.can_read_body:
            self.bodies[line] = await request.json()

    def _next(self) -> int:
        self._ids += 1
        return self._ids

    def _page(self, request: web.Request, rows: list[dict[str, Any]]) -> web.Response:
        size = min(int(request.query.get("limit", "50")), self.page_size)
        start = int(request.query.get("cursor", "0"))
        items = rows[start : start + size]
        page: dict[str, Any] = {
            "items": items,
            "has_more": start + size < len(rows),
        }
        if page["has_more"]:
            page["next_cursor"] = str(start + size)
        return web.json_response(page)

    async def _list_limits(self, request: web.Request) -> web.Response:
        await self._record(request)
        if self.lists_before_create:
            self.lists_before_create -= 1
            return self._page(request, [])
        return self._page(
            request, [r for r in self.limits if matches(r, request.query)]
        )

    async def _create_limit(self, request: web.Request) -> web.Response:
        await self._record(request)
        sent = self.body("POST", request.path)
        for row in self.limits:
            if same_subject(row, sent) and row["interval"] == sent["interval"]:
                return refused(
                    409,
                    "limit_exists",
                    f"the subject has a {sent['interval']} limit at that level already: "
                    f"limit {row['id']}; change it instead",
                )
        if sent.get("limit_micros", 0) > 10_000_000_000:
            return refused(
                400, "limit_above_parent", "above the organization's monthly limit"
            )
        fields = {k: v for k, v in sent.items() if k != "limit_micros"}
        row = self.add_limit(limit_micros=sent["limit_micros"], **fields)
        return web.json_response(status=201, data=row)

    async def _update_limit(self, request: web.Request) -> web.Response:
        await self._record(request)
        row = self._limit(int(request.match_info["id"]))
        if row is None:
            return refused(404, "limit_not_found", "there is no such limit")
        row.update(self.body("PATCH", request.path))
        row["effective_limit_micros"] = row["limit_micros"]
        return web.json_response(row)

    async def _delete_limit(self, request: web.Request) -> web.Response:
        await self._record(request)
        row = self._limit(int(request.match_info["id"]))
        if row is None:
            return refused(404, "limit_not_found", "there is no such limit")
        self.limits.remove(row)
        return web.Response(status=204)

    async def _create_grant(self, request: web.Request) -> web.Response:
        await self._record(request)
        sent = self.body("POST", request.path)
        if not any(same_subject(row, sent) for row in self.limits):
            return refused(
                404, "no_limit", "the subject has no spend limit to grant past"
            )
        fields = {k: v for k, v in sent.items() if k != "amount_micros"}
        row = self.add_grant(amount_micros=sent["amount_micros"], **fields)
        return web.json_response(status=201, data=row)

    async def _list_grants(self, request: web.Request) -> web.Response:
        await self._record(request)
        return self._page(
            request, [r for r in self.grants if matches(r, request.query)]
        )

    async def _revoke(self, request: web.Request) -> web.Response:
        await self._record(request)
        [row] = [g for g in self.grants if g["id"] == int(request.match_info["id"])]
        row.update(state="revoked", revoked_at=WHEN, revoked_by="api_key:key")
        return web.json_response(row)

    async def _usage(self, request: web.Request) -> web.Response:
        await self._record(request)
        lists = {name: request.query.getall(name, []) for name in IDS}
        count = max([len(values) for values in lists.values()] + [1])
        answered = []
        for i in range(count):
            named = {
                name: values[0] if len(values) == 1 else values[i]
                for name, values in lists.items()
                if values
            }
            answered.append(usage(request.query["subject"], **named))
        return web.json_response(answered)

    async def _reset(self, request: web.Request) -> web.Response:
        await self._record(request)
        sent = self.body("POST", request.path)
        if not any(same_subject(row, sent) for row in self.limits):
            return refused(404, "no_limit", "the subject has no spend limit to reset")
        named = {name: sent[name] for name in IDS if name in sent}
        return web.json_response(usage(sent["subject"], **named))

    async def _teams(self, request: web.Request) -> web.Response:
        await self._record(request)
        app_id = request.query.get("app_id")
        return web.json_response(
            [t for t in self.teams if app_id is None or app_id in t["apps"]]
        )

    async def _account(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(self.account)

    async def _credits(self, request: web.Request) -> web.Response:
        await self._record(request)
        return self._page(request, self.credits)

    async def _spend(self, request: web.Request) -> web.Response:
        await self._record(request)
        return web.json_response(self.buckets)

    def _limit(self, id: int) -> Optional[dict[str, Any]]:
        return next((row for row in self.limits if row["id"] == id), None)


@asynccontextmanager
async def served() -> AsyncIterator[SpendRouter]:
    """A stand-in listening on localhost for as long as the block runs."""
    fake = SpendRouter()
    server = TestServer(fake.app())
    await server.start_server()
    fake.url = str(server.make_url("")).rstrip("/")
    try:
        yield fake
    finally:
        await server.close()


def matches(row: dict[str, Any], query: MultiDictProxy[str]) -> bool:
    """Whether a listed row passes the filters a list was sent with."""
    for name in ("subject", "level", "interval", "state", *IDS):
        if name in query and row.get(name) != query[name]:
            return False
    return True


def same_subject(row: dict[str, Any], sent: dict[str, Any]) -> bool:
    return row["subject"] == sent["subject"] and all(
        row.get(name) == sent.get(name) for name in IDS
    )


def limit(id: int, **fields: Any) -> dict[str, Any]:
    level = fields.pop("level", None) or implied_level(fields)
    micros = fields.pop("limit_micros", 5_000_000)
    return {
        "id": id,
        "level": level,
        "interval": "daily",
        "limit_micros": micros,
        "effective_limit_micros": micros,
        "updated_by": "api_key:key",
        "created_at": WHEN,
        "updated_at": WHEN,
        **fields,
    }


def implied_level(fields: dict[str, Any]) -> str:
    """The level the router fills in for a limit sent without one."""
    if fields.get("subject") == "api_key" and fields.get("api_key"):
        return "api_key"
    if fields.get("subject") == "user" and fields.get("user_id"):
        return "user"
    if fields.get("app_id") or fields.get("subject") in APP_SUBJECTS:
        return "app"
    return "organization"


def grant(id: int, **fields: Any) -> dict[str, Any]:
    amount = fields.pop("amount_micros", 1_000_000)
    return {
        "id": id,
        "amount_micros": amount,
        "used_micros": 0,
        "remaining_micros": amount,
        "state": "live",
        "reason": "support",
        "created_by": "api_key:key",
        "created_at": WHEN,
        **fields,
    }


def usage(subject: str, **named: str) -> dict[str, Any]:
    return {
        "subject": subject,
        **named,
        "windows": [
            {
                "interval": "daily",
                "window_start": WHEN,
                "window_end": WINDOW_END,
                "spent_micros": 1_250_000,
                "baseline_micros": 0,
                "room_micros": 3_750_000,
                "on_grant": False,
                "limit": {
                    "id": 1,
                    "level": "organization",
                    "limit_micros": 5_000_000,
                    "effective_limit_micros": 5_000_000,
                },
            }
        ],
        "grants": [],
    }


def account(**fields: Any) -> dict[str, Any]:
    return {
        "organization_id": "1181507",
        "status": "active",
        "tier": "verified",
        "balance_micros": 9_000_000,
        "granted_micros": 10_000_000,
        "spent_micros": 1_000_000,
        "balance_state": "ok",
        "balance_warning_micros": 5_000_000,
        "balance_warning_source": "tier",
        "limits": {},
        **fields,
    }
