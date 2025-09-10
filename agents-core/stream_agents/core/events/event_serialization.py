"""
Event serialization utilities for GetStream AI plugins.

This module provides functions for serializing and deserializing events
for storage or transmission.
"""

import json
from typing import Dict, Any, List
from datetime import datetime

from .events import BaseEvent, EventType, create_event


def serialize_event(event: BaseEvent) -> Dict[str, Any]:
    """Serialize an event to a dictionary."""
    data = event.to_dict()

    # Handle special cases for non-serializable fields
    if hasattr(event, "error") and event.error:
        data["error"] = {
            "type": type(event.error).__name__,
            "message": str(event.error),
            "args": getattr(event.error, "args", []),
        }

    if hasattr(event, "audio_data") and event.audio_data:
        # Don't serialize large audio data, just metadata
        data["audio_data"] = {"size_bytes": len(event.audio_data), "type": "bytes"}

    return data


def serialize_events(events: List[BaseEvent]) -> str:
    """Serialize a list of events to JSON string."""
    serialized_events = [serialize_event(event) for event in events]
    return json.dumps(serialized_events, indent=2, default=str)


def deserialize_event(data: Dict[str, Any]) -> BaseEvent:
    """Deserialize an event from a dictionary."""
    event_type_str = data.get("event_type")
    if not event_type_str:
        raise ValueError("Event data missing event_type")

    try:
        event_type = EventType(event_type_str)
    except ValueError:
        raise ValueError(f"Unknown event type: {event_type_str}")

    # Remove fields that shouldn't be passed to constructor
    constructor_data = data.copy()
    constructor_data.pop("event_type", None)

    # Handle special fields
    if "timestamp" in constructor_data and isinstance(
        constructor_data["timestamp"], str
    ):
        constructor_data["timestamp"] = datetime.fromisoformat(
            constructor_data["timestamp"]
        )

    # Handle error reconstruction (simplified)
    if "error" in constructor_data and isinstance(constructor_data["error"], dict):
        error_info = constructor_data["error"]
        constructor_data["error"] = Exception(
            error_info.get("message", "Unknown error")
        )

    # Remove audio data placeholder
    if "audio_data" in constructor_data and isinstance(
        constructor_data["audio_data"], dict
    ):
        constructor_data.pop("audio_data")

    return create_event(event_type, **constructor_data)
