"""Twilio call registry for tracking active calls."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .media_stream import TwilioMediaStream

logger = logging.getLogger(__name__)


@dataclass
class TwilioCall:
    """
    Represents an active Twilio call session.
    
    Stores all webhook form data from Twilio along with associated
    connections and timestamps.
    """
    call_sid: str
    form_data: dict[str, Any] = field(default_factory=dict)
    twilio_stream: Optional["TwilioMediaStream"] = None
    stream_call: Optional[Any] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    @property
    def from_number(self) -> Optional[str]:
        """Get the caller's phone number."""
        return self.form_data.get("From")
    
    @property
    def to_number(self) -> Optional[str]:
        """Get the called phone number."""
        return self.form_data.get("To")
    
    @property
    def call_status(self) -> Optional[str]:
        """Get the call status."""
        return self.form_data.get("CallStatus")
    
    @property
    def caller_city(self) -> Optional[str]:
        """Get the caller's city."""
        return self.form_data.get("CallerCity")
    
    @property
    def caller_state(self) -> Optional[str]:
        """Get the caller's state."""
        return self.form_data.get("CallerState")
    
    @property
    def caller_country(self) -> Optional[str]:
        """Get the caller's country."""
        return self.form_data.get("CallerCountry")
    
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
    
    def create(self, call_sid: str, form_data: Optional[dict[str, Any]] = None) -> TwilioCall:
        """
        Create and register a new TwilioCall.
        
        Args:
            call_sid: The Twilio CallSid.
            form_data: All form data from the Twilio webhook.
            
        Returns:
            The created TwilioCall instance.
        """
        if form_data is None:
            form_data = {}
        call = TwilioCall(call_sid=call_sid, form_data=form_data)
        self._calls[call_sid] = call
        logger.info(f"TwilioCallRegistry: Created call {call_sid} from {call.from_number} to {call.to_number}")
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

