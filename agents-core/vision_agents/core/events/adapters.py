from dataclasses import asdict
from typing import Any, Protocol

from .base import AgentConnectionEvent


class SupportsCustomEvent(Protocol):
    async def send_custom_event(self, data: dict[str, Any]) -> None: ...


class EdgeCustomEventOutboundAdapter:
    """Bridge outbound AgentConnectionEvent messages to edge custom events."""

    def __init__(self, edge: SupportsCustomEvent):
        self._edge = edge

    async def deliver(self, event: AgentConnectionEvent) -> None:
        await self._edge.send_custom_event(asdict(event))
