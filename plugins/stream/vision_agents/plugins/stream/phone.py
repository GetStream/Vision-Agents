import logging
from typing import Optional

from vision_agents.core.telephony import OutboundCall, PlacedCall

from ._backend import Backend
from ._generated.api.default import (
    attach_phone_number,
    buy_phone_number,
    list_phone_numbers,
    place_phone_call,
    release_phone_number,
    search_phone_numbers,
)
from ._generated.models import (
    AttachedNumber,
    AttachNumberRequest,
    BuyNumberRequest,
    Error,
    NumberSearchResult,
    PhoneCapability,
    PhoneNumber,
    PhoneNumberType,
    PlaceCallRequest,
    PlaceCallRequestCustom,
    PlaceCallRequestHeaders,
)
from ._generated.types import UNSET, Unset

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

    async def search(
        self,
        country: str,
        vendor: Optional[str] = None,
        area_code: Optional[str] = None,
        contains: Optional[str] = None,
        prefix: Optional[str] = None,
        locality: Optional[str] = None,
        administrative_area: Optional[str] = None,
        number_type: Optional[PhoneNumberType] = None,
        features: Optional[list[PhoneCapability]] = None,
        limit: int = 10,
    ) -> NumberSearchResult:
        """Search for numbers to buy, at one vendor or all of them.

        Naming a vendor searches only that one. Leaving it out asks every vendor that has
        credentials, and merges what they offer cheapest first. A vendor whose API cannot
        express one of the filters is reported in `skipped` rather than asked without it.

        Args:
            country: The country to search in.
            vendor: Search only this vendor. Defaults to every vendor with credentials.
            area_code: Restrict to this area code.
            contains: Restrict to numbers containing this substring.
            prefix: Restrict to numbers starting with this prefix.
            locality: Restrict to this locality.
            administrative_area: Restrict to this administrative area.
            number_type: What kind of number to search for.
            features: Capabilities the number must have.
            limit: Maximum numbers to return.

        Returns:
            The numbers offered, cheapest first, and which vendors were skipped and why.

        Raises:
            RuntimeError: If the router refused the search, saying why.
        """
        result = await search_phone_numbers.asyncio(
            client=self.backend.client(),
            country=country,
            vendor=vendor if vendor is not None else UNSET,
            area_code=area_code if area_code is not None else UNSET,
            contains=contains if contains is not None else UNSET,
            prefix=prefix if prefix is not None else UNSET,
            locality=locality if locality is not None else UNSET,
            administrative_area=administrative_area
            if administrative_area is not None
            else UNSET,
            number_type=number_type if number_type is not None else UNSET,
            features=features if features is not None else UNSET,
            limit=limit,
        )
        if isinstance(result, Error):
            raise RuntimeError(result.error)
        if result is None:
            raise RuntimeError("the router did not answer with a search result")
        return result

    async def buy(
        self,
        vendor: str,
        e164: str,
        country: Optional[str] = None,
    ) -> PhoneNumber:
        """Buy a number, which starts its monthly charge.

        Args:
            vendor: Who to buy the number from.
            e164: The number to buy.
            country: The country the number was offered from. Most vendors buy by number
                alone; the few that buy out of a country's inventory need this.

        Returns:
            The bought number.

        Raises:
            RuntimeError: If the router refused the purchase, saying why.
        """
        request = BuyNumberRequest(vendor=vendor, e164=e164)
        if country is not None:
            request.country = country

        bought = await buy_phone_number.asyncio(
            client=self.backend.client(), body=request
        )
        if isinstance(bought, Error):
            raise RuntimeError(bought.error)
        if bought is None:
            raise RuntimeError("the router did not answer with a bought number")

        logger.info("bought %s from %s", bought.e164, vendor)
        return bought

    async def attach(
        self,
        e164: str,
        call_id: Optional[str] = None,
        call_type: Optional[str] = None,
        allowed_ips: Optional[list[str]] = None,
    ) -> AttachedNumber:
        """Point a number at a Stream call.

        Creates the SIP inbound trunk and routing rule and tells the vendor to send calls
        there. This is what turns a bought number into one that reaches an agent.

        Args:
            e164: The number to attach.
            call_id: The call every caller joins. Omit to give each caller their own call,
                named after the number they rang.
            call_type: The Stream call type. Omit for "agent".
            allowed_ips: The vendor's signalling addresses, as IPs or CIDR blocks.

        Returns:
            Where the trunk and routing rule were created, and the SIP URI the vendor sends
            calls to.

        Raises:
            RuntimeError: If the router refused to attach the number, saying why.
        """
        body: AttachNumberRequest | Unset = UNSET
        if call_id is not None or call_type is not None or allowed_ips is not None:
            request = AttachNumberRequest()
            if call_id is not None:
                request.call_id = call_id
            if call_type is not None:
                request.call_type = call_type
            if allowed_ips is not None:
                request.allowed_ips = allowed_ips
            body = request

        attached = await attach_phone_number.asyncio(
            e164=e164, client=self.backend.client(), body=body
        )
        if isinstance(attached, Error):
            raise RuntimeError(attached.error)
        if attached is None:
            raise RuntimeError("the router did not answer with an attached number")

        logger.info("attached %s, sip_uri %s", e164, attached.sip_uri)
        return attached

    async def release(self, e164: str) -> None:
        """Give a number back, which stops its monthly charge.

        Args:
            e164: The number to release.

        Raises:
            RuntimeError: If the router refused to release the number, saying why.
        """
        released = await release_phone_number.asyncio(
            e164=e164, client=self.backend.client()
        )
        if isinstance(released, Error):
            raise RuntimeError(released.error)

        logger.info("released %s", e164)


def _or_empty(value: object) -> str:
    """Read an optional string the router may have left out."""
    if value is UNSET or value is None:
        return ""
    return str(value)
