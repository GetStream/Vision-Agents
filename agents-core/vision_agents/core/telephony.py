"""What an agent needs of a telephony provider to ring somebody.

The provider itself lives in a plugin, because the phone paths belong to the acceleration
backend rather than to the core. This is the shape the core asks for, so `Agent` can place a
call without importing anything optional.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol


class _Joined(Protocol):
    async def wait_for_participant(self, timeout: Optional[float] = None) -> None: ...


@dataclass
class OutboundCall:
    """The terms a call is placed on.

    Attributes:
        from_: One of your own numbers, which is what the person sees.
        to: Who to call.
        call_id: The call the answered leg joins, and so the one the agent has to be in.
        call_type: The call type. Empty means "default".
        ring_timeout: How long to ring before giving up. None leaves the vendor's default,
            which is long enough to reach voicemail.
        initial_digits: Pressed once the person answers, for reaching an extension behind
            a menu, e.g. "ww1234#".
        headers: Carried to the person's leg as custom SIP headers. Only some vendors can
            express these, and one that cannot refuses the call rather than dropping them.
        custom: Put on the call, where the agent in it can read it.
    """

    from_: str
    to: str
    call_id: str = ""
    call_type: str = ""
    ring_timeout: Optional[float] = None
    initial_digits: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    custom: dict[str, str] = field(default_factory=dict)


@dataclass
class PlacedCall:
    """A call that is on its way, and where to meet it.

    Attributes:
        vendor_call_id: Identifies the ringing leg at the vendor.
        status: The vendor's own word for where the call is, e.g. "queued".
        vendor: Who is placing it.
        call_id: The call the answered leg is routed into. An agent that is not in it
            hears nothing when the person picks up.
        call_type: The type of that call.
    """

    vendor_call_id: str
    status: str
    vendor: str = ""
    call_id: str = ""
    call_type: str = ""


@dataclass
class InboundCall:
    """Somebody has rung one of your numbers and is waiting in a call.

    The call already exists: the caller reached it over SIP before anything here knew about
    it, which is why this names a call to join rather than one to create.

    Attributes:
        call_id: The call the caller is in. An agent that joins anything else hears silence.
        call_type: The type of that call.
        called_number: The number that was rung, which is how an agent serving several
            numbers knows which line this is.
        caller_number: Who is calling. Empty when they had not finished joining yet, which
            says nothing about whether they are there.
        custom: Whatever was put on the call.
        at: When the call arrived, so a call just handed over can be told from one that
            waited.
    """

    call_id: str
    call_type: str = "default"
    called_number: str = ""
    caller_number: str = ""
    custom: dict[str, str] = field(default_factory=dict)
    at: Optional[datetime] = None
    _joined: Optional[_Joined] = field(default=None, repr=False, compare=False)

    async def wait_for_phone_participant(self, timeout: Optional[float] = None) -> None:
        """Wait until the caller is in the call.

        Join first: there is nobody to wait for until the agent is in the call they
        arrived on.
        """
        if self._joined is None:
            raise RuntimeError("join the call before waiting for the caller")
        await self._joined.wait_for_participant(timeout=timeout)


CallContext = InboundCall


class Telephony(Protocol):
    """Somewhere to place an outbound call."""

    async def place(self, call: OutboundCall) -> PlacedCall:
        """Ring somebody and bridge the answered leg into a call."""
        ...
