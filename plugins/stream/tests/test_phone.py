from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from vision_agents.core.telephony import OutboundCall
from vision_agents.plugins import stream


class Router:
    """A stand-in for the acceleration router, serving the call it is asked to place.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive.
    """

    def __init__(self):
        self.placed: Optional[dict[str, Any]] = None
        self.transferred: Optional[dict[str, Any]] = None
        self.searched: Optional[dict[str, Any]] = None
        self.bought: Optional[dict[str, Any]] = None
        self.attached: Optional[dict[str, Any]] = None
        self.attached_e164: Optional[str] = None
        self.released_e164: Optional[str] = None
        self.url = ""
        self.refuse = ""

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/phone/calls", self._place)
        app.router.add_post("/v1/phone/calls/transfer", self._transfer)
        app.router.add_get("/v1/phone/numbers/available", self._search)
        app.router.add_post("/v1/phone/numbers", self._buy)
        app.router.add_post("/v1/phone/numbers/{e164}/attach", self._attach)
        app.router.add_delete("/v1/phone/numbers/{e164}", self._release)
        return app

    async def _place(self, request: web.Request) -> web.Response:
        self.placed = await request.json()
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.json_response(
            status=202,
            data={
                "vendor_call_id": "CA123",
                "status": "queued",
                "vendor": "twilio",
                "call_id": self.placed.get("call_id", "call-made-up"),
                "call_type": self.placed.get("call_type", "default"),
            },
        )

    async def _transfer(self, request: web.Request) -> web.Response:
        self.transferred = await request.json()
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.json_response(
            status=202,
            data={
                "vendor_call_id": "CA456",
                "status": "queued",
                "vendor": "twilio",
                "call_id": self.transferred["call_id"],
                "call_type": self.transferred.get("call_type", "agent"),
            },
        )

    async def _search(self, request: web.Request) -> web.Response:
        self.searched = dict(request.query)
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.json_response(
            status=200,
            data={
                "numbers": [
                    {
                        "e164": "+15125551234",
                        "vendor": "telnyx",
                        "country": "US",
                        "capabilities": ["voice", "sms"],
                    }
                ],
                "skipped": [
                    {"vendor": "twilio", "reason": "cannot search by area_code"}
                ],
            },
        )

    async def _buy(self, request: web.Request) -> web.Response:
        self.bought = await request.json()
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.json_response(
            status=201,
            data={
                "e164": self.bought["e164"],
                "vendor": self.bought["vendor"],
                "country": self.bought.get("country", "US"),
                "capabilities": ["voice"],
                "monthly_cost_micros": 1_000_000,
                "purchased_at": "2026-01-01T00:00:00Z",
            },
        )

    async def _attach(self, request: web.Request) -> web.Response:
        self.attached_e164 = request.match_info["e164"]
        self.attached = await request.json() if request.body_exists else {}
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.json_response(
            status=200,
            data={
                "trunk_id": "trunk-1",
                "route_id": "route-1",
                "sip_uri": "sip:trunk@sip.stream-io-api.com",
            },
        )

    async def _release(self, request: web.Request) -> web.Response:
        self.released_e164 = request.match_info["e164"]
        if self.refuse:
            return web.json_response(status=400, data={"error": self.refuse})
        return web.Response(status=204)


class TestPhone:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    def phone(self, router: Router) -> stream.Phone:
        return stream.Phone(url=router.url, customer_id="acme")

    async def test_every_term_of_the_call_reaches_the_router(
        self, router: Router, phone: stream.Phone
    ):
        await phone.place(
            OutboundCall(
                from_="+17195551234",
                to="+13035559876",
                call_id="support-line",
                call_type="livestream",
                ring_timeout=20.0,
                initial_digits="ww1234#",
                headers={"X-Ticket": "42"},
                custom={"reason": "callback"},
            )
        )

        assert router.placed is not None
        assert router.placed["from"] == "+17195551234"
        assert router.placed["to"] == "+13035559876"
        assert router.placed["call_id"] == "support-line"
        assert router.placed["call_type"] == "livestream"
        assert router.placed["ring_timeout_seconds"] == 20
        assert router.placed["initial_digits"] == "ww1234#"
        assert router.placed["headers"] == {"X-Ticket": "42"}
        assert router.placed["custom"] == {"reason": "callback"}

    async def test_a_call_with_no_terms_asks_for_none_of_them(
        self, router: Router, phone: stream.Phone
    ):
        # Sending a zero ring timeout would ask every vendor to give up immediately.
        await phone.place(OutboundCall(from_="+17195551234", to="+13035559876"))

        assert router.placed is not None
        assert set(router.placed) == {"from", "to"}

    async def test_what_comes_back_names_the_call_the_agent_has_to_join(
        self, router: Router, phone: stream.Phone
    ):
        placed = await phone.place(
            OutboundCall(
                from_="+17195551234", to="+13035559876", call_id="support-line"
            )
        )

        assert placed.vendor_call_id == "CA123"
        assert placed.status == "queued"
        assert placed.vendor == "twilio"
        assert placed.call_id == "support-line"
        assert placed.call_type == "default"

    async def test_a_refused_call_says_why(self, router: Router, phone: stream.Phone):
        # A vendor that cannot express one of the terms refuses the call, and the reason
        # is the only way a caller learns which term it was.
        router.refuse = "phone: bird cannot place a call with ring_timeout"

        with pytest.raises(RuntimeError, match="ring_timeout"):
            await phone.place(
                OutboundCall(from_="+17195551234", to="+13035559876", ring_timeout=20.0)
            )

    async def test_transfer_reaches_the_router(
        self, router: Router, phone: stream.Phone
    ):
        placed = await phone.transfer(
            from_="+17195551234",
            to="+13035559876",
            call_id="support-line",
            call_type="livestream",
        )

        assert router.transferred is not None
        assert router.transferred["from"] == "+17195551234"
        assert router.transferred["to"] == "+13035559876"
        assert router.transferred["call_id"] == "support-line"
        assert router.transferred["call_type"] == "livestream"
        assert placed.vendor_call_id == "CA456"
        assert placed.status == "queued"
        assert placed.vendor == "twilio"
        assert placed.call_id == "support-line"
        assert placed.call_type == "livestream"

    async def test_transfer_with_no_call_type_asks_for_none(
        self, router: Router, phone: stream.Phone
    ):
        await phone.transfer(
            from_="+17195551234", to="+13035559876", call_id="support-line"
        )

        assert router.transferred is not None
        assert set(router.transferred) == {"from", "to", "call_id"}

    async def test_a_refused_transfer_says_why(
        self, router: Router, phone: stream.Phone
    ):
        router.refuse = "phone: bird cannot transfer a call"

        with pytest.raises(RuntimeError, match="cannot transfer"):
            await phone.transfer(
                from_="+17195551234", to="+13035559876", call_id="support-line"
            )

    async def test_search_returns_both_offered_and_skipped_vendors(
        self, router: Router, phone: stream.Phone
    ):
        # Dropping `skipped` would answer a search for eight vendors as if only two exist.
        result = await phone.search(country="US", area_code="512")

        assert router.searched is not None
        assert router.searched["country"] == "US"
        assert router.searched["area_code"] == "512"
        assert len(result.numbers) == 1
        assert result.numbers[0].e164 == "+15125551234"
        assert result.numbers[0].vendor == "telnyx"
        assert len(result.skipped) == 1
        assert result.skipped[0].vendor == "twilio"
        assert result.skipped[0].reason == "cannot search by area_code"

    async def test_buy_returns_the_bought_number(
        self, router: Router, phone: stream.Phone
    ):
        bought = await phone.buy(vendor="telnyx", e164="+15125551234", country="US")

        assert router.bought == {
            "vendor": "telnyx",
            "e164": "+15125551234",
            "country": "US",
        }
        assert bought.e164 == "+15125551234"
        assert bought.vendor == "telnyx"

    async def test_attach_returns_the_trunk_route_and_sip_uri(
        self, router: Router, phone: stream.Phone
    ):
        attached = await phone.attach(
            "+15125551234", call_id="support-line", call_type="livestream"
        )

        assert router.attached_e164 == "+15125551234"
        assert router.attached == {
            "call_id": "support-line",
            "call_type": "livestream",
        }
        assert attached.trunk_id == "trunk-1"
        assert attached.route_id == "route-1"
        assert attached.sip_uri == "sip:trunk@sip.stream-io-api.com"

    async def test_release_completes(self, router: Router, phone: stream.Phone):
        await phone.release("+15125551234")

        assert router.released_e164 == "+15125551234"

    async def test_a_refused_search_says_why(self, router: Router, phone: stream.Phone):
        router.refuse = "phone: bird cannot search by administrative_area"

        with pytest.raises(RuntimeError, match="administrative_area"):
            await phone.search(country="US", administrative_area="TX")

    def test_public_types_are_the_generated_ones(self):
        from vision_agents.plugins.stream._generated.models import (
            AttachedNumber,
            NumberSearchResult,
            PhoneNumber,
            SkippedVendor,
        )

        assert stream.PhoneNumber is PhoneNumber
        assert stream.AttachedNumber is AttachedNumber
        assert stream.NumberSearchResult is NumberSearchResult
        assert stream.SkippedVendor is SkippedVendor
