import asyncio
from typing import List

import pytest

from getstream.plugins.common import STS


class DummySTS(STS):
    """Minimal concrete implementation used only for testing the base class."""

    async def fake_connect(self):
        self._is_connected = True
        self.emit("connected")

    async def fake_disconnect(self):
        self._is_connected = False
        self.emit("disconnected")


@pytest.mark.asyncio
async def test_event_emission_and_state():
    sts = DummySTS()
    received: List[str] = []

    @sts.on("connected")  # type: ignore[arg-type]
    async def _on_connected():
        received.append("connected")

    @sts.on("disconnected")  # type: ignore[arg-type]
    async def _on_disconnected():
        received.append("disconnected")

    # Initially not connected
    assert sts.is_connected is False

    # Simulate connect lifecycle
    await sts.fake_connect()
    await asyncio.sleep(0)  # allow event loop to run callbacks
    assert sts.is_connected is True
    assert "connected" in received

    # Simulate disconnect lifecycle
    await sts.fake_disconnect()
    await asyncio.sleep(0)
    assert sts.is_connected is False
    assert received == ["connected", "disconnected"]
