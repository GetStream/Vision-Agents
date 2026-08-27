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
        self.url = ""
        self.refuse = ""

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/v1/phone/calls", self._place)
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
