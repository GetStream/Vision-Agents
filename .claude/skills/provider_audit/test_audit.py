"""Tests for audit.py against a real local HTTP server standing in for each vendor.

Run from the repo root with `uv run --no-project --with pytest pytest -c /dev/null
--confcutdir .claude/skills/provider_audit .claude/skills/provider_audit`.
"""

import base64
import json
import socket
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import audit
import pytest

KEY = "sk-test-0123456789abcdef0123456789abcdef"


class Request:
    """What a check sent: its path, query and headers (names lower case)."""

    def __init__(self, path: str, query: dict[str, str], headers: dict[str, str]):
        self.path, self.query, self.headers = path, query, headers


class Vendor:
    """A local server that answers each path with a fixed status and body.

    It records every request, so a test can assert the documented auth and query.
    """

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, bytes]] = {}
        self.requests: list[Request] = []
        routes, requests = self.routes, self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path, _, query = self.path.partition("?")
                headers = {k.lower(): v for k, v in self.headers.items()}
                requests.append(
                    Request(path, dict(urllib.parse.parse_qsl(query)), headers)
                )
                status, body = routes.get(path, (404, b"{}"))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(
            target=self.server.serve_forever, args=(0.01,), daemon=True
        ).start()

    def answer(self, path: str, body: object, status: int = 200) -> None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.routes[path] = (status, raw)


@contextmanager
def dead_server(answer: Callable[[socket.socket], None]) -> Iterator[str]:
    """A raw socket server that hands each connection to answer, for broken replies."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)

    def serve() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            with conn:
                answer(conn)

    threading.Thread(target=serve, daemon=True).start()
    yield f"http://127.0.0.1:{server.getsockname()[1]}"
    server.close()


def month_start() -> datetime:
    now = datetime.now(UTC)
    return datetime(now.year, now.month, 1, tzinfo=UTC)


@pytest.fixture
def vendor() -> Iterator[Vendor]:
    server = Vendor()
    yield server
    server.server.shutdown()


@pytest.fixture
def local(vendor: Vendor) -> list[audit.Check]:
    """One check, OpenRouter's, pointed at the local server, keyed by TEST_AUDIT_KEY."""
    return [("Local", ("TEST_AUDIT_KEY",), lambda k: audit.openrouter(k, vendor.url))]


def credits(total: object, usage: object) -> dict[str, object]:
    return {"data": {"total_credits": total, "total_usage": usage}}


class TestAudit:
    def test_balance_is_credits_minus_usage(self, vendor: Vendor) -> None:
        vendor.answer("/api/v1/credits", credits(200, 81.25))
        assert audit.openrouter(KEY, vendor.url) == [
            (118.75, "USD", "credits bought minus used, whole account")
        ]

    @pytest.mark.parametrize(
        ("status", "body", "error"),
        [
            (401, b'{"error": "bad key"}', "HTTP 401"),
            (402, b"Payment required", "HTTP 402"),
            (200, b"<html>login</html>", "not JSON"),
            (200, b"[1, 2]", "expected a JSON object, got list"),
            (200, b"{}", "no data.total_credits"),
            (200, b'{"data": null}', "no data.total_credits"),
            (200, json.dumps(credits("lots", 1)).encode(), "not a number"),
            (200, json.dumps(credits(True, 1)).encode(), "not a number"),
            (200, json.dumps(credits(None, 1)).encode(), "not a number"),
            (200, json.dumps(credits({"usd": 5}, 1)).encode(), "not a number"),
        ],
    )
    def test_anything_but_a_balance_is_an_error_row(
        self,
        vendor: Vendor,
        local: list[audit.Check],
        monkeypatch: pytest.MonkeyPatch,
        status: int,
        body: bytes,
        error: str,
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/api/v1/credits", body, status)
        [row] = audit.audit("now", local, {})
        assert row["left"] is None
        assert str(row["note"]).startswith("error (TEST_AUDIT_KEY): ")
        assert error in str(row["note"])

    def test_numeric_strings_are_balances(self, vendor: Vendor) -> None:
        vendor.answer("/api/v1/credits", credits("50.5", "0.5"))
        assert audit.openrouter(KEY, vendor.url)[0][0] == 50.0

    def test_unreachable_vendor_is_an_error(self) -> None:
        with pytest.raises(audit.AuditError, match="unreachable"):
            audit.openrouter(KEY, "http://127.0.0.1:9")

    def test_error_text_never_holds_the_key(self, vendor: Vendor) -> None:
        vendor.answer("/api/v1/credits", f'{{"error": "bad key {KEY}"}}'.encode(), 401)
        with pytest.raises(audit.AuditError) as error:
            audit.openrouter(KEY, vendor.url)
        assert KEY not in str(error.value)
        assert "<key>" in str(error.value)

    def test_error_text_hides_tokens_and_emails(self, vendor: Vendor) -> None:
        token = "tok_" + "x1" * 20
        body = f'{{"error": "session {token} of ops@example.com"}}'.encode()
        vendor.answer("/api/v1/credits", body, 403)
        with pytest.raises(audit.AuditError) as error:
            audit.openrouter(KEY, vendor.url)
        assert token not in str(error.value)
        assert "ops@example.com" not in str(error.value)
        assert "session <redacted> of <email>" in str(error.value)

    def test_a_page_that_is_not_json_never_shows_the_key(self, vendor: Vendor) -> None:
        vendor.answer("/api/v1/credits", f"<p>{KEY}</p>".encode())
        with pytest.raises(audit.AuditError, match="not JSON") as error:
            audit.openrouter(KEY, vendor.url)
        assert KEY not in str(error.value)
        assert "<key>" in str(error.value)

    @pytest.mark.parametrize(
        "reply",
        [
            pytest.param(b"", id="closed-without-reply"),
            pytest.param(
                b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"a":1',
                id="short-body",
            ),
        ],
    )
    def test_a_broken_reply_is_an_error_row(
        self, monkeypatch: pytest.MonkeyPatch, reply: bytes
    ) -> None:
        def answer(conn: socket.socket) -> None:
            conn.recv(65536)
            conn.sendall(reply)

        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        with dead_server(answer) as url:
            check = ("Local", ("TEST_AUDIT_KEY",), lambda k: audit.openrouter(k, url))
            [row] = audit.audit("now", [check], {})
        assert str(row["note"]).startswith("error (TEST_AUDIT_KEY): unreachable")

    def test_a_vendor_that_never_answers_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        silent = threading.Event()

        def answer(conn: socket.socket) -> None:
            conn.recv(65536)
            silent.wait(5)

        monkeypatch.setattr(audit, "TIMEOUT", 0.2)
        with dead_server(answer) as url:
            with pytest.raises(audit.AuditError, match="unreachable: timed out"):
                audit.openrouter(KEY, url)
            silent.set()

    def test_unset_key_is_skipped(
        self, local: list[audit.Check], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_AUDIT_KEY", raising=False)
        [row] = audit.audit("now", local, {})
        assert row["left"] is None
        assert row["note"] == "skipped: TEST_AUDIT_KEY not set"

    def test_one_row_per_key_in_the_pool(
        self,
        vendor: Vendor,
        local: list[audit.Check],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", "a,b")
        vendor.answer("/api/v1/credits", credits(10, 1))
        rows = audit.audit("now", local, {})
        assert [r["vendor"] for r in rows] == [
            "Local (TEST_AUDIT_KEY #1)",
            "Local (TEST_AUDIT_KEY #2)",
        ]

    def test_an_account_with_no_balances_says_so(
        self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/v1/projects", {"projects": []})

        def check(k: str) -> list[audit.Reading]:
            return audit.deepgram(k, vendor.url)

        [row] = audit.audit("now", [("D", ("TEST_AUDIT_KEY",), check)], {})
        assert (row["left"], row["note"]) == (None, "no balances in the answer")

    def test_a_list_that_is_not_a_list_is_an_error(self, vendor: Vendor) -> None:
        vendor.answer("/v1/projects", {"projects": None})
        with pytest.raises(audit.AuditError, match="projects is None, not a list"):
            audit.deepgram(KEY, vendor.url)

    def test_dashboard_only_accounts_get_a_row(self) -> None:
        [row] = audit.audit("now", [], {"Exa": "https://dashboard.exa.ai/billing"})
        assert row == {
            "vendor": "Exa",
            "type": "core",
            "left": None,
            "unit": "",
            "note": "unknown (dashboard only)",
            "source": "https://dashboard.exa.ai/billing",
            "checked_at": "now",
        }

    def test_first_env_set_wins(
        self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_AUDIT_ADMIN", raising=False)
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/api/v1/credits", credits(10, 1))

        def check(k: str) -> list[audit.Reading]:
            return audit.openrouter(k, vendor.url)

        [row] = audit.audit(
            "now", [("V", ("TEST_AUDIT_ADMIN", "TEST_AUDIT_KEY"), check)], {}
        )
        assert row["left"] == 9.0
        monkeypatch.setenv("TEST_AUDIT_ADMIN", "admin")
        vendor.answer("/api/v1/credits", b"{}", 401)
        [row] = audit.audit(
            "now", [("V", ("TEST_AUDIT_ADMIN", "TEST_AUDIT_KEY"), check)], {}
        )
        assert "error (TEST_AUDIT_ADMIN)" in str(row["note"])


class TestKeys:
    def test_reads_the_pool_as_keypool_does(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_POOL", " a, b,a,, ")
        monkeypatch.setenv("TEST_POOL_2", "c")
        monkeypatch.setenv("TEST_POOL_10", "e")
        monkeypatch.setenv("TEST_POOL_1", "d,b")
        monkeypatch.setenv("TEST_POOL_01", "not numbered")
        monkeypatch.setenv("TEST_POOL_X", "not numbered")
        monkeypatch.setenv("TEST_POOLED", "another name")
        assert audit.keys("TEST_POOL") == ["a", "b", "d", "c", "e"]

    def test_numbered_keys_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_ONLY", raising=False)
        monkeypatch.setenv("TEST_ONLY_1", "k1")
        assert audit.keys("TEST_ONLY") == ["k1"]


ROWS: list[audit.Row] = [
    {"vendor": "Low", "left": 3.21, "unit": "USD", "note": ""},
    {"vendor": "Edge", "left": 20.0, "unit": "USD", "note": ""},
    {"vendor": "High", "left": 118.0, "unit": "USD", "note": ""},
    {"vendor": "Chars", "left": 5.0, "unit": "characters", "note": ""},
    {"vendor": "Spent", "left": 1.0, "unit": "USD spent this month", "note": ""},
    {"vendor": "Broken", "left": None, "unit": "", "note": "error: HTTP 401"},
]


class TestThreshold:
    def test_only_rows_under_the_min_of_their_unit(self) -> None:
        low = audit.below(ROWS, {"usd": 20.0})
        assert [r["vendor"] for r in low] == ["Low"]

    def test_units_match_without_case(self) -> None:
        low = audit.below(ROWS, {"characters": 10.0})
        assert [r["vendor"] for r in low] == ["Chars"]

    @pytest.mark.parametrize(
        ("text", "parsed"), [("USD=20", ("usd", 20.0)), (" usd = 2.5", ("usd", 2.5))]
    )
    def test_parse_min(self, text: str, parsed: tuple[str, float]) -> None:
        assert audit.parse_min(text) == parsed

    @pytest.mark.parametrize("text", ["USD", "USD=", "USD=lots"])
    def test_parse_min_refuses(self, text: str) -> None:
        with pytest.raises(Exception, match="want UNIT=AMOUNT"):
            audit.parse_min(text)

    @pytest.mark.parametrize(
        ("left", "args", "code"),
        [
            (3.21, ["--min", "USD=20"], 1),
            (20.0, ["--min", "USD=20"], 0),
            (3.21, ["--min", "credits=20"], 0),
            (3.21, [], 0),
        ],
    )
    def test_exit_code(
        self,
        vendor: Vendor,
        local: list[audit.Check],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        left: float,
        args: list[str],
        code: int,
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/api/v1/credits", credits(left, 0))
        assert audit.main(args, local, {}) == code
        out = capsys.readouterr().out
        assert f"Local | {left:.2f} | USD |" in out
        assert ("BELOW: Local" in out) == (code == 1)

    def test_an_error_never_fails_the_run(
        self,
        vendor: Vendor,
        local: list[audit.Check],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/api/v1/credits", b"{}", 401)
        assert audit.main(["--min", "USD=20"], local, {}) == 0
        assert "Local | error |" in capsys.readouterr().out


class TestVendorTypes:
    """Core vendors fail the run under --min; connector vendors are only shown."""

    @pytest.fixture
    def two(self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch) -> list[audit.Check]:
        """A connector check listed first, then a core one, both on the local server."""
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)

        def check(k: str) -> list[audit.Reading]:
            return audit.openrouter(k, vendor.url)

        return [
            ("Phone", ("TEST_AUDIT_KEY",), check),
            ("Model", ("TEST_AUDIT_KEY",), check),
        ]

    def test_rows_carry_their_type(
        self, two: list[audit.Check], vendor: Vendor
    ) -> None:
        vendor.answer("/api/v1/credits", credits(10, 1))
        rows = audit.audit("now", two, {"Pager": "https://x"}, frozenset({"Phone"}))
        assert [(r["vendor"], r["type"]) for r in rows] == [
            ("Phone", "connector"),
            ("Model", "core"),
            ("Pager", "core"),
        ]

    def test_telnyx_and_twilio_are_connectors(self) -> None:
        rows = audit.audit("now", [], {"Telnyx": "", "Twilio": "", "Exa": ""})
        assert [r["type"] for r in rows] == ["connector", "connector", "core"]

    def test_core_rows_first_then_connectors_under_a_heading(
        self,
        two: list[audit.Check],
        vendor: Vendor,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        vendor.answer("/api/v1/credits", credits(10, 1))
        audit.main([], two, {}, frozenset({"Phone"}))
        lines = capsys.readouterr().out.splitlines()
        heading = "Connector vendors (bring your own; never fail the run)"
        model = next(i for i, x in enumerate(lines) if x.startswith("Model |"))
        phone = next(i for i, x in enumerate(lines) if x.startswith("Phone |"))
        assert model < lines.index(heading) < phone

    def test_no_connector_heading_without_connector_rows(
        self,
        two: list[audit.Check],
        vendor: Vendor,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        vendor.answer("/api/v1/credits", credits(10, 1))
        audit.main([], two, {}, frozenset())
        assert "Connector vendors" not in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("connectors", "code", "line"),
        [
            (frozenset({"Phone"}), 1, "BELOW: Model has 3.00 USD"),
            (frozenset({"Model"}), 0, "BELOW (connector): Model has 3.00 USD"),
        ],
    )
    def test_only_a_low_core_balance_fails_the_run(
        self,
        vendor: Vendor,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        connectors: frozenset[str],
        code: int,
        line: str,
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer("/api/v1/credits", credits(3, 0))

        def check(k: str) -> list[audit.Reading]:
            return audit.openrouter(k, vendor.url)

        model: list[audit.Check] = [("Model", ("TEST_AUDIT_KEY",), check)]
        assert audit.main(["--min", "USD=20"], model, {}, connectors) == code
        out = capsys.readouterr().out.splitlines()
        assert [x for x in out if x.startswith("BELOW")] == [line]


BEARER = {"authorization": f"Bearer {KEY}"}


class TestChecks:
    """Each check sends its vendor's documented request and reads its answer."""

    @pytest.mark.parametrize(
        ("fn", "routes", "auth", "query", "readings"),
        [
            (
                audit.openrouter,
                {"/api/v1/credits": credits(30, 7.5)},
                BEARER,
                {},
                [(22.5, "USD", "credits bought minus used, whole account")],
            ),
            (
                audit.tavily,
                {
                    "/usage": {
                        "account": {
                            "plan_limit": 250,
                            "plan_usage": 40,
                            "current_plan": "test-plan",
                        }
                    }
                },
                BEARER,
                {},
                [(210.0, "credits", "plan test-plan, this cycle")],
            ),
            (
                audit.telnyx,
                {
                    "/v2/balance": {
                        "data": {"available_credit": "400.00", "currency": "USD"}
                    }
                },
                BEARER,
                {},
                [(400.0, "USD", "balance + credit limit")],
            ),
            (
                audit.elevenlabs,
                {
                    "/v1/user/subscription": {
                        "character_limit": 100,
                        "character_count": 40,
                        "tier": "pro",
                        "next_character_count_reset_unix": 1,
                    }
                },
                {"xi-api-key": KEY},
                {},
                [(60.0, "characters", "tier pro, resets at unix 1")],
            ),
            (
                audit.deepgram,
                {
                    "/v1/projects": {
                        "projects": [
                            {"project_id": "p1", "name": "ops@example.com"},
                            {"project_id": "p2", "name": "b"},
                        ]
                    },
                    "/v1/projects/p1/balances": {
                        "balances": [{"amount": 12.5, "units": "usd"}]
                    },
                    "/v1/projects/p2/balances": {"balances": []},
                },
                {"authorization": f"Token {KEY}"},
                {},
                [(12.5, "usd", "project <email>")],
            ),
            (
                audit.fish,
                {"/wallet/self/api-credit": {"credit": "7.25"}},
                BEARER,
                {},
                [(7.25, "credits", "API credit")],
            ),
            (
                audit.fal,
                {
                    "/v1/account/billing": {
                        "credits": {"current_balance": 24.5, "currency": "USD"}
                    }
                },
                {"authorization": f"Key {KEY}"},
                {"/v1/account/billing": {"expand": "credits"}},
                [(24.5, "USD", "")],
            ),
            (
                audit.ollama,
                {
                    "/api/balance": {
                        "included": {"balance_usd": 1.5},
                        "purchased": {"balance_usd": 2},
                    }
                },
                BEARER,
                {},
                [(3.5, "USD", "included + purchased credits")],
            ),
            (
                audit.baseten,
                {
                    "/v1/billing/usage_summary": {
                        "dedicated_usage": {"total": "10.5"},
                        "training_usage": None,
                        "model_apis_usage": {"total": "2"},
                    }
                },
                BEARER,
                {
                    "/v1/billing/usage_summary": {
                        "start_date": month_start().isoformat()
                    }
                },
                [
                    (
                        12.5,
                        "USD spent this month",
                        "postpaid: dedicated + training + APIs",
                    )
                ],
            ),
            (
                audit.openai,
                {
                    "/v1/organization/costs": {
                        "data": [
                            {"results": [{"amount": {"value": 1.5}}]},
                            {"results": []},
                        ]
                    }
                },
                BEARER,
                {
                    "/v1/organization/costs": {
                        "start_time": str(int(month_start().timestamp()))
                    }
                },
                [(1.5, "USD spent this month", "credit balance: dashboard only")],
            ),
            (
                audit.anthropic,
                {
                    "/v1/organizations/cost_report": {
                        "data": [{"results": [{"amount": "123.45"}, {"amount": "100"}]}]
                    }
                },
                {"x-api-key": KEY, "anthropic-version": "2023-06-01"},
                {
                    "/v1/organizations/cost_report": {
                        "starting_at": month_start().strftime("%Y-%m-%dT%H:%M:%SZ")
                    }
                },
                [(2.2345, "USD spent this month", "credit balance: dashboard only")],
            ),
        ],
    )
    def test_sends_the_documented_request_and_reads_the_answer(
        self,
        vendor: Vendor,
        fn: Callable[[str, str], list[audit.Reading]],
        routes: dict[str, object],
        auth: dict[str, str],
        query: dict[str, dict[str, str]],
        readings: list[audit.Reading],
    ) -> None:
        for path, body in routes.items():
            vendor.answer(path, body)
        expected: list[object] = [
            (pytest.approx(left), unit, note) for left, unit, note in readings
        ]
        assert fn(KEY, vendor.url) == expected
        assert sorted(r.path for r in vendor.requests) == sorted(routes)
        for request in vendor.requests:
            assert auth.items() <= request.headers.items(), request.path
            assert query.get(request.path, {}).items() <= request.query.items()

    @pytest.mark.parametrize(
        ("fn", "path", "body"),
        [
            (audit.baseten, "/v1/billing/usage_summary", {"training_usage": None}),
            (audit.openai, "/v1/organization/costs", {"data": []}),
        ],
    )
    def test_no_spend_prints_zero_not_unknown(
        self,
        vendor: Vendor,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        fn: Callable[[str, str], list[audit.Reading]],
        path: str,
        body: dict[str, object],
    ) -> None:
        monkeypatch.setenv("TEST_AUDIT_KEY", KEY)
        vendor.answer(path, body)

        def check(k: str) -> list[audit.Reading]:
            return fn(k, vendor.url)

        audit.main([], [("Spend", ("TEST_AUDIT_KEY",), check)], {})
        assert "Spend | 0.00 | USD spent this month |" in capsys.readouterr().out

    def test_twilio_needs_its_account_sid(
        self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        with pytest.raises(audit.AuditError, match="TWILIO_ACCOUNT_SID"):
            audit.twilio(KEY, vendor.url)
        monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC1")
        vendor.answer(
            "/2010-04-01/Accounts/AC1/Balance.json",
            {"balance": "6.5", "currency": "USD"},
        )
        assert audit.twilio(KEY, vendor.url) == [(6.5, "USD", "")]
        basic = base64.b64encode(f"AC1:{KEY}".encode()).decode()
        assert vendor.requests[-1].headers["authorization"] == f"Basic {basic}"

    def test_xai_reads_the_team_of_the_api_key(
        self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-key")
        vendor.answer("/v1/api-key", {"team_id": "t1"})
        # The doc: total is USD cents, and a purchase is negative.
        vendor.answer(
            "/v1/billing/teams/t1/prepaid/balance", {"total": {"val": "-500"}}
        )
        assert audit.xai(KEY, vendor.url, vendor.url) == [
            (5.0, "USD", "prepaid credit")
        ]
        team, balance = vendor.requests
        assert team.headers["authorization"] == "Bearer xai-key"
        assert balance.headers["authorization"] == f"Bearer {KEY}"

    def test_xai_without_an_api_key_says_so(
        self, vendor: Vendor, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        vendor.answer("/v1/api-key", b'{"error": "bad"}', 401)
        with pytest.raises(audit.AuditError, match="^XAI_API_KEY is not set"):
            audit.xai(KEY, vendor.url, vendor.url)
        assert vendor.requests == []

    def test_every_check_names_its_doc(self) -> None:
        for vendor, _, fn in audit.CHECKS:
            assert str(fn.__doc__).startswith("https://"), vendor
