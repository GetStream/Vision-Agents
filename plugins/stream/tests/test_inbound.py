import asyncio
import os
from typing import AsyncIterator

import httpx
import pytest
from vision_agents.core.telephony import InboundCall, OutboundCall
from vision_agents.plugins import stream

URL_ENV = "STREAM_ACCELERATION_URL"
CUSTOMER_ENV = "STREAM_ACCELERATION_CUSTOMER_ID"
PUBLIC_ENV = "ROUTER_PUBLIC_URL"
NUMBER_ENV = "INBOUND_NUMBER"

# A real call has to be placed, ring, and be answered by Stream's SIP before the router
# hears about it. None of that is fast.
RINGING = 150.0


def digits(number: str) -> str:
    """The number without the punctuation vendors disagree about."""
    return "".join(character for character in number if character.isdigit())


@pytest.mark.integration
class TestInboundCalling:
    """Rings a number this customer holds and asserts a waiting worker was given the call.

    This is the whole path with nothing stood in for: a vendor rings a vendor, Stream's SIP
    answers on the inbound trunk, the router hears the webhook, and the call comes back down
    the dispatch socket. It needs a router that Stream can reach, so it is skipped unless
    ROUTER_PUBLIC_URL is set, and it needs `cmd/phone hooks -url $ROUTER_PUBLIC_URL` to have
    been run once against the app.
    """

    @pytest.fixture(autouse=True)
    def router(self) -> None:
        for name in (URL_ENV, CUSTOMER_ENV, PUBLIC_ENV, NUMBER_ENV):
            if not os.getenv(name):
                pytest.skip(f"{name} is not set")

    @pytest.fixture
    def called(self) -> str:
        """The attached number that gets rung."""
        return os.environ[NUMBER_ENV]

    @pytest.fixture
    async def phone(self) -> stream.Phone:
        return stream.Phone()

    @pytest.fixture
    async def caller(self, phone: stream.Phone, called: str) -> str:
        """A second held number to ring from.

        A held number rather than a handset, so the test places the call itself instead of
        asking somebody to pick up a phone.
        """
        held = await phone.numbers()
        wanted = digits(called)
        others = [
            number.e164
            for number in held
            if digits(number.e164) != wanted and not number.released_at
        ]
        if not others:
            pytest.skip(
                f"only one number is held, so there is nothing to ring {called} from"
            )
        return others[0]

    @pytest.fixture
    async def attached(self, called: str) -> None:
        """The number, connected to a trunk and a routing rule.

        Attaching is idempotent at the router, so a number already in service is left where
        it is and this only makes the test independent of whether anybody attached it.
        """
        async with httpx.AsyncClient(
            base_url=os.environ[URL_ENV],
            headers={"X-Customer-Id": os.environ[CUSTOMER_ENV]},
            timeout=60.0,
        ) as client:
            response = await client.post(f"/v1/phone/numbers/{called}/attach", json={})
        assert response.status_code == 200, response.text
        assert response.json()["sip_uri"], "the vendor has nowhere to send the call"

    @pytest.fixture
    async def waiting(self) -> AsyncIterator[asyncio.Queue]:
        """A worker connected to the router and waiting for calls."""
        arrived: asyncio.Queue[InboundCall] = asyncio.Queue()
        worker = stream.StreamDispatch(capacity=1)

        @worker.wait_for_call()
        async def answer(call: InboundCall) -> None:
            await arrived.put(call)

        running = asyncio.create_task(worker.run())

        async def registered() -> None:
            while not worker.worker_id:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(registered(), 30.0)
        yield arrived
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    async def test_ringing_an_attached_number_hands_the_call_to_a_waiting_worker(
        self,
        phone: stream.Phone,
        called: str,
        caller: str,
        attached: None,
        waiting: asyncio.Queue,
    ):
        placed = await phone.place(
            OutboundCall(from_=caller, to=called, ring_timeout=45.0)
        )
        assert placed.vendor_call_id, "the vendor never accepted the outgoing leg"

        async def rung() -> InboundCall:
            # The outgoing leg is a call of its own, and it starts a session too. Only the
            # one that landed on the rung number is this test's.
            while True:
                call = await waiting.get()
                if digits(call.called_number) == digits(called):
                    return call

        call = await asyncio.wait_for(rung(), RINGING)

        # The routing rule names the call after the number that was rung, and that name is
        # the only thing an agent has to go on when it joins.
        assert call.call_id == "phone-" + called
        assert digits(call.caller_number) == digits(caller)
