"""Telnyx call registry for tracking active calls."""

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .media_stream import TelnyxMediaStream

logger = logging.getLogger(__name__)


@dataclass
class TelnyxCall:
    """
    Represents an active Telnyx call session.
    """

    call_control_id: str
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    webhook_data: Optional[dict[str, Any]] = None
    telnyx_stream: Optional["TelnyxMediaStream"] = None
    stream_call: Optional[Any] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    _prepare_task: Optional[asyncio.Task] = field(default=None, repr=False)

    @property
    def payload(self) -> dict[str, Any]:
        return (
            self.webhook_data.get("data", {}).get("payload", {})
            if self.webhook_data
            else {}
        )

    @property
    def from_number(self) -> Optional[str]:
        return self.payload.get("from")

    @property
    def to_number(self) -> Optional[str]:
        return self.payload.get("to")

    @property
    def call_status(self) -> Optional[str]:
        return self.payload.get("state")

    def end(self):
        """Mark the call as ended."""
        self.ended_at = datetime.utcnow()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get call duration in seconds, or None if still active."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    async def await_prepare(self) -> Any:
        """
        Wait for the prepare task to complete and return its result.
        """
        if self._prepare_task is None:
            return None
        return await self._prepare_task


class TelnyxCallRegistry:
    """
    In-memory registry for active Telnyx calls.
    """

    def __init__(self):
        self._calls: dict[str, TelnyxCall] = {}

    def create(
        self,
        call_control_id: str,
        webhook_data: Optional[dict[str, Any]] = None,
        prepare: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None,
    ) -> TelnyxCall:
        call = TelnyxCall(call_control_id=call_control_id, webhook_data=webhook_data)

        if prepare is not None:
            call._prepare_task = asyncio.create_task(prepare())

        self._calls[call_control_id] = call
        logger.info(
            "TelnyxCallRegistry: Created call %s",
            call.call_control_id,
        )
        return call

    def get(self, call_control_id: str) -> Optional[TelnyxCall]:
        return self._calls.get(call_control_id)

    def require(self, call_control_id: str) -> TelnyxCall:
        call = self._calls.get(call_control_id)
        if not call:
            raise ValueError(f"Unknown call_control_id: {call_control_id}")
        return call

    def validate(self, call_control_id: str, token: str) -> TelnyxCall:
        call = self._calls.get(call_control_id)
        if not call:
            raise ValueError(f"Unknown call_control_id: {call_control_id}")
        if not secrets.compare_digest(call.token, token):
            raise ValueError(f"Invalid token for call_control_id: {call_control_id}")
        return call

    def remove(self, call_control_id: str) -> Optional[TelnyxCall]:
        call = self._calls.pop(call_control_id, None)
        if call:
            call.end()
            duration = call.duration_seconds
            logger.info(
                "TelnyxCallRegistry: Removed call %s (duration: %.1fs)",
                call_control_id,
                duration,
            )
        return call

    def list_active(self) -> list[TelnyxCall]:
        return [call for call in self._calls.values() if call.ended_at is None]

    def __len__(self) -> int:
        return len(self._calls)
