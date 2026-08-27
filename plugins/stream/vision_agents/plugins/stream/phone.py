import logging
from typing import Optional

from vision_agents.core.telephony import OutboundCall, PlacedCall

from ._backend import Backend
from ._generated.api.default import list_phone_numbers, place_phone_call
from ._generated.models import (
    Error,
    PhoneNumber,
    PlaceCallRequest,
    PlaceCallRequestCustom,
    PlaceCallRequestHeaders,
)
from ._generated.types import UNSET

logger = logging.getLogger(__name__)


class Phone:
    """The telephony half of the acceleration backend: calls to place on your numbers.

    Stream's SIP is inbound only, so a placed call is the vendor ringing somebody and
    bridging the answered leg into a Stream call. The agent has to be in that call to be
    heard, which is what `Agent.outbound_call` arranges.

    Example:
        ```python
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=agent_user,
            llm=stream.Accelerated(config="john"),
            phone=stream.Phone(),
        )
        async with agent.outbound_call(from_=held, to=person):
            await agent.finish()
        ```
    """

    def __init__(
        self,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
    ):
        """Reach the telephony paths on a router.

        Args:
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            customer_id: Who the calls are billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
        """
        self.backend = Backend(url=url, customer_id=customer_id)

    async def place(self, call: OutboundCall) -> PlacedCall:
        """Ring somebody and bridge the answered leg into a call.

        Returns as soon as the vendor has accepted the call, which is before it rings.

        Args:
            call: Who to call and on what terms.

        Returns:
            The ringing leg, and the call its answer is routed into.

        Raises:
            RuntimeError: If the router refused the call, saying why. A vendor whose API
                cannot express one of the terms asked for refuses rather than dropping it.
        """
        request = PlaceCallRequest(from_=call.from_, to=call.to)
        if call.call_id:
            request.call_id = call.call_id
        if call.call_type:
            request.call_type = call.call_type
        if call.ring_timeout is not None:
            request.ring_timeout_seconds = int(call.ring_timeout)
        if call.initial_digits:
            request.initial_digits = call.initial_digits
        if call.headers:
            headers = PlaceCallRequestHeaders()
            headers.additional_properties = dict(call.headers)
            request.headers = headers
        if call.custom:
            custom = PlaceCallRequestCustom()
            custom.additional_properties = dict(call.custom)
            request.custom = custom

        placed = await place_phone_call.asyncio(
            client=self.backend.client(), body=request
        )
        if isinstance(placed, Error):
            raise RuntimeError(placed.error)
        if placed is None:
            raise RuntimeError("the router did not answer with a placed call")

        logger.info("calling %s, vendor call %s", call.to, placed.vendor_call_id)
        return PlacedCall(
            vendor_call_id=placed.vendor_call_id,
            status=placed.status,
            vendor=_or_empty(placed.vendor),
            call_id=_or_empty(placed.call_id),
            call_type=_or_empty(placed.call_type),
        )

    async def numbers(self) -> list[PhoneNumber]:
        """The numbers this customer holds, which are the ones a call can be placed from."""
        held = await list_phone_numbers.asyncio(client=self.backend.client())
        if isinstance(held, Error):
            raise RuntimeError(held.error)
        if held is None:
            raise RuntimeError("the router did not answer with any numbers")
        return held


def _or_empty(value: object) -> str:
    """Read an optional string the router may have left out."""
    if value is UNSET or value is None:
        return ""
    return str(value)
