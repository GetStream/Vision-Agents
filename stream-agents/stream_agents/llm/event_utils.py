"""
Event utilities and registry for GetStream AI plugins.

This module provides utilities for event handling, filtering, serialization,
and debugging across all plugin types.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict, deque

from .events import (
    BaseEvent, EventType
)

logger = logging.getLogger(__name__)


class EventFilter:
    """Filter events based on various criteria."""

    def __init__(
        self,
        event_types: Optional[List[EventType]] = None,
        session_ids: Optional[List[str]] = None,
        plugin_names: Optional[List[str]] = None,
        time_window_ms: Optional[int] = None,
        min_confidence: Optional[float] = None
    ):
        """
        Initialize event filter.

        Args:
            event_types: List of event types to include
            session_ids: List of session IDs to include
            plugin_names: List of plugin names to include
            time_window_ms: Only include events within this time window (ms from now)
            min_confidence: Minimum confidence threshold for applicable events
        """
        self.event_types = set(event_types) if event_types else None
        self.session_ids = set(session_ids) if session_ids else None
        self.plugin_names = set(plugin_names) if plugin_names else None
        self.time_window_ms = time_window_ms
        self.min_confidence = min_confidence

    def matches(self, event: BaseEvent) -> bool:
        """Check if an event matches the filter criteria."""

        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check session ID
        if self.session_ids and event.session_id not in self.session_ids:
            return False

        # Check plugin name
        if self.plugin_names and event.plugin_name not in self.plugin_names:
            return False

        # Check time window
        if self.time_window_ms:
            event_time_ms = event.timestamp.timestamp() * 1000
            current_time_ms = time.time() * 1000
            if current_time_ms - event_time_ms > self.time_window_ms:
                return False

        # Check confidence (strict filtering - events without confidence are excluded)
        if self.min_confidence is not None:
            if not hasattr(event, 'confidence') or event.confidence is None:
                return False  # Exclude events without confidence data
            if event.confidence < self.min_confidence:
                return False  # Exclude low confidence events

        return True


class EventRegistry:
    """Registry for tracking and analyzing events across all plugins."""

    def __init__(self, max_events: int = 10000):
        """
        Initialize event registry.

        Args:
            max_events: Maximum number of events to keep in memory
        """
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[EventType, int] = defaultdict(int)
        self.session_events: Dict[str, List[BaseEvent]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.listeners: Dict[EventType, List[Callable]] = defaultdict(list)

    def register_event(self, event: BaseEvent):
        """Register a new event in the registry."""
        self.events.append(event)
        self.event_counts[event.event_type] += 1

        if event.session_id:
            self.session_events[event.session_id].append(event)

        # Track errors separately
        if "ERROR" in event.event_type.value.upper():
            error_key = f"{event.plugin_name}_{event.event_type.value}"
            self.error_counts[error_key] += 1

        # Notify listeners
        for listener in self.listeners.get(event.event_type, []):
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")

    def add_listener(self, event_type: EventType, listener: Callable[[BaseEvent], None]):
        """Add a listener for a specific event type."""
        self.listeners[event_type].append(listener)

    def remove_listener(self, event_type: EventType, listener: Callable):
        """Remove a listener for a specific event type."""
        if listener in self.listeners[event_type]:
            self.listeners[event_type].remove(listener)

    def get_events(
        self,
        filter_criteria: Optional[EventFilter] = None,
        limit: Optional[int] = None
    ) -> List[BaseEvent]:
        """Get events matching the filter criteria."""
        events = list(self.events)

        if filter_criteria:
            events = [e for e in events if filter_criteria.matches(e)]

        if limit:
            events = events[-limit:]  # Get most recent events

        return events

    def get_session_events(self, session_id: str) -> List[BaseEvent]:
        """Get all events for a specific session."""
        return self.session_events.get(session_id, [])

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors across all plugins."""
        total_events = len(self.events)
        total_errors = sum(self.error_counts.values())

        return {
            "total_events": total_events,
            "total_errors": total_errors,
            "error_rate": total_errors / total_events if total_events > 0 else 0,
            "error_breakdown": dict(self.error_counts),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about events."""
        total_events = len(self.events)

        # Calculate event type distribution
        event_distribution = {
            event_type.value: count
            for event_type, count in self.event_counts.items()
        }

        # Calculate session statistics
        session_count = len(self.session_events)
        avg_events_per_session = (
            total_events / session_count if session_count > 0 else 0
        )

        # Calculate time-based statistics
        if self.events:
            oldest_event = min(self.events, key=lambda e: e.timestamp)
            newest_event = max(self.events, key=lambda e: e.timestamp)
            time_span_ms = (newest_event.timestamp - oldest_event.timestamp).total_seconds() * 1000
            events_per_second = total_events / (time_span_ms / 1000) if time_span_ms > 0 else 0
        else:
            time_span_ms = 0
            events_per_second = 0

        return {
            "total_events": total_events,
            "event_distribution": event_distribution,
            "session_count": session_count,
            "avg_events_per_session": avg_events_per_session,
            "time_span_ms": time_span_ms,
            "events_per_second": events_per_second,
            "error_summary": self.get_error_summary()
        }

    def clear(self):
        """Clear all events from the registry."""
        self.events.clear()
        self.event_counts.clear()
        self.session_events.clear()
        self.error_counts.clear()



class EventLogger:
    """Enhanced logging for events with structured output."""

    def __init__(self, logger_name: str = __name__):
        self.logger = logging.getLogger(logger_name)
        self.registry = EventRegistry()

    def log_event(self, event: BaseEvent, log_level: int = logging.INFO):
        """Log an event with structured information."""
        self.registry.register_event(event)

        # Create structured log entry
        log_data = {
            "event_type": event.event_type.value,
            "event_id": event.event_id,
            "session_id": event.session_id,
            "plugin_name": event.plugin_name,
            "timestamp": event.timestamp.isoformat(),
        }

        # Add event-specific information
        if hasattr(event, 'text'):
            log_data["text_length"] = len(event.text) if event.text else 0
            log_data["text_preview"] = event.text[:100] if event.text else ""

        if hasattr(event, 'confidence'):
            log_data["confidence"] = event.confidence

        if hasattr(event, 'processing_time_ms'):
            log_data["processing_time_ms"] = event.processing_time_ms

        if hasattr(event, 'audio_data') and event.audio_data is not None:
            # Handle numpy arrays and other array-like objects
            if hasattr(event.audio_data, '__len__'):
                log_data["audio_bytes"] = len(event.audio_data)
            else:
                log_data["audio_bytes"] = 0

        if hasattr(event, 'error'):
            log_data["error_message"] = str(event.error)
            log_data["error_type"] = type(event.error).__name__

        # Log with appropriate level
        message = f"{event.event_type.value}: {event.plugin_name or 'unknown'}"
        self.logger.log(log_level, message, extra=log_data)

    def get_registry(self) -> EventRegistry:
        """Get the event registry for analysis."""
        return self.registry


# Global event registry instance
global_event_registry = EventRegistry()

# Global event logger instance
global_event_logger = EventLogger("getstream.plugins.events")


def register_global_event(event: BaseEvent):
    """Register an event in the global registry."""
    global_event_registry.register_event(event)
    global_event_logger.log_event(event)


def get_global_registry() -> EventRegistry:
    """Get the global event registry."""
    return global_event_registry


def get_global_logger() -> EventLogger:
    """Get the global event logger."""
    return global_event_logger


__all__ = [
    "EventFilter",
    "EventRegistry",
    "EventLogger",
    "global_event_registry",
    "global_event_logger",
    "register_global_event",
    "get_global_registry",
    "get_global_logger",
]
