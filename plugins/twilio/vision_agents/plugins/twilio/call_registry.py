"""Twilio call registry for tracking active calls."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .media_stream import TwilioMediaStream
    from .models import CallWebhookInput

logger = logging.getLogger(__name__)


@dataclass
class TwilioCall:
    """
    Represents an active Twilio call session.
    
    Stores the webhook data from Twilio along with associated
    connections and timestamps.
    """
    call_sid: str
    webhook_data: Optional["CallWebhookInput"] = None
    twilio_stream: Optional["TwilioMediaStream"] = None
    stream_call: Optional[Any] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    @property
    def from_number(self) -> Optional[str]:
        """Get the caller's phone number."""
        return self.webhook_data.from_number if self.webhook_data else None
    
    @property
    def to_number(self) -> Optional[str]:
        """Get the called phone number."""
        return self.webhook_data.to if self.webhook_data else None
    
    @property
    def call_status(self) -> Optional[str]:
        """Get the call status."""
        return self.webhook_data.call_status if self.webhook_data else None
    
    @property
    def caller(self) -> Optional[str]:
        """Get the caller's phone number (alias)."""
        return self.webhook_data.caller if self.webhook_data else None
    
    @property
    def caller_city(self) -> Optional[str]:
        """Get the caller's city."""
        return self.webhook_data.caller_city if self.webhook_data else None
    
    @property
    def caller_state(self) -> Optional[str]:
        """Get the caller's state."""
        return self.webhook_data.caller_state if self.webhook_data else None
    
    @property
    def caller_country(self) -> Optional[str]:
        """Get the caller's country."""
        return self.webhook_data.caller_country if self.webhook_data else None
    
    @property
    def direction(self) -> Optional[str]:
        """Get the call direction (inbound/outbound)."""
        return self.webhook_data.direction if self.webhook_data else None
    
    @property
    def stir_verstat(self) -> Optional[str]:
        """Get the STIR/SHAKEN verification status."""
        return self.webhook_data.stir_verstat if self.webhook_data else None
    
    def end(self):
        """Mark the call as ended."""
        self.ended_at = datetime.utcnow()
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get call duration in seconds, or None if still active."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class TwilioCallRegistry:
    """
    In-memory registry for active Twilio calls.
    
    Tracks calls by their Twilio CallSid and provides methods
    for creating, retrieving, and removing calls.
    """
    
    def __init__(self):
        self._calls: dict[str, TwilioCall] = {}
    
    def create(self, call_id: str, webhook_data: Optional["CallWebhookInput"] = None) -> TwilioCall:
        """
        Create and register a new TwilioCall.
        
        Args:
            call_id: The unique identifier for this call.
            webhook_data: Optional parsed webhook data from Twilio (for inbound calls).
            
        Returns:
            The created TwilioCall instance.
        """
        call = TwilioCall(call_sid=call_id, webhook_data=webhook_data)
        self._calls[call_id] = call
        logger.info(f"TwilioCallRegistry: Created call {call.call_sid} from {call.from_number} to {call.to_number}")
        return call
    
    def get(self, call_sid: str) -> Optional[TwilioCall]:
        """
        Get a TwilioCall by its SID.
        
        Args:
            call_sid: The Twilio CallSid.
            
        Returns:
            The TwilioCall if found, None otherwise.
        """
        return self._calls.get(call_sid)
    
    def require(self, call_sid: str) -> TwilioCall:
        """
        Get a TwilioCall by its SID, raising if not found.
        
        Args:
            call_sid: The Twilio CallSid.
            
        Returns:
            The TwilioCall.
            
        Raises:
            ValueError: If no call with the given SID exists.
        """
        call = self._calls.get(call_sid)
        if not call:
            raise ValueError(f"Unknown call_sid: {call_sid}")
        return call
    
    def remove(self, call_sid: str) -> Optional[TwilioCall]:
        """
        Remove a TwilioCall from the registry.
        
        Args:
            call_sid: The Twilio CallSid.
            
        Returns:
            The removed TwilioCall if found, None otherwise.
        """
        call = self._calls.pop(call_sid, None)
        if call:
            call.end()
            duration = call.duration_seconds
            logger.info(f"TwilioCallRegistry: Removed call {call_sid} (duration: {duration:.1f}s)")
        return call
    
    def list_active(self) -> list[TwilioCall]:
        """
        List all active (not ended) calls.
        
        Returns:
            List of active TwilioCall instances.
        """
        return [c for c in self._calls.values() if c.ended_at is None]
    
    def __len__(self) -> int:
        """Return the number of registered calls."""
        return len(self._calls)
