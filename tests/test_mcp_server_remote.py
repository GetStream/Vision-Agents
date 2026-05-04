"""Regression tests for ``MCPServerRemote`` lifecycle.

The MCP transport stack (``streamablehttp_client`` + ``ClientSession``) is
built on anyio cancel scopes that require ``__aenter__`` and ``__aexit__`` to
run in the same asyncio task. An earlier version of ``MCPServerRemote``
entered the contexts in the caller's connect-task and exited them in a
different disconnect-task (typical for shielded teardown), which left a
half-cancelled scope that pegged the event loop on ``_deliver_cancellation``.

These tests pin the contract: regardless of which task drives ``connect()``
and ``disconnect()``, the underlying transports' ``__aenter__`` / ``__aexit__``
must run on the same asyncio task.
"""

import asyncio
import logging
from typing import Any, Optional
from unittest.mock import AsyncMock


from vision_agents.core.mcp import mcp_server_remote
from vision_agents.core.mcp.mcp_server_remote import MCPServerRemote


class _TaskTrackingAsyncCM:
    """Async context manager that records which asyncio task entered/exited
    and how many times each happened."""

    def __init__(self, on_enter: Any) -> None:
        self.enter_count = 0
        self.exit_count = 0
        self.enter_task: Optional[asyncio.Task] = None
        self.exit_task: Optional[asyncio.Task] = None
        self._on_enter = on_enter

    async def __aenter__(self) -> Any:
        self.enter_count += 1
        self.enter_task = asyncio.current_task()
        return self._on_enter

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit_count += 1
        self.exit_task = asyncio.current_task()
        return None


def _install_transport_mocks(
    monkeypatch,
) -> tuple[_TaskTrackingAsyncCM, _TaskTrackingAsyncCM]:
    """Install task-tracking mocks for ``streamablehttp_client`` and
    ``ClientSession``. Returns the two CM instances for assertions."""
    fake_session = AsyncMock()
    fake_session.initialize = AsyncMock()
    transport_cm = _TaskTrackingAsyncCM(
        on_enter=("read-stub", "write-stub", lambda: "session-id-stub"),
    )
    session_cm = _TaskTrackingAsyncCM(on_enter=fake_session)
    monkeypatch.setattr(
        mcp_server_remote,
        "streamablehttp_client",
        lambda *args, **kwargs: transport_cm,
    )
    monkeypatch.setattr(
        mcp_server_remote,
        "ClientSession",
        lambda read, write: session_cm,
    )
    return transport_cm, session_cm


async def test_transport_contexts_enter_and_exit_in_same_task(monkeypatch):
    """``connect()`` in one task and ``disconnect()`` in another must still
    leave the underlying MCP transport contexts entered and exited from a
    single asyncio task."""

    transport_cm, session_cm = _install_transport_mocks(monkeypatch)
    server = MCPServerRemote(url="http://test.example/mcp")

    connect_task = asyncio.current_task()
    await server.connect()

    disconnect_task: Optional[asyncio.Task] = None

    async def disconnect_in_other_task() -> None:
        nonlocal disconnect_task
        disconnect_task = asyncio.current_task()
        await server.disconnect()

    await asyncio.create_task(disconnect_in_other_task())

    # Both transport contexts must have been entered and exited.
    assert transport_cm.enter_task is not None
    assert transport_cm.exit_task is not None
    assert session_cm.enter_task is not None
    assert session_cm.exit_task is not None

    # The invariant: enter and exit happen on the same asyncio task.
    assert transport_cm.enter_task is transport_cm.exit_task, (
        "streamablehttp_client must enter and exit on the same asyncio task "
        f"(entered={transport_cm.enter_task!r}, exited={transport_cm.exit_task!r})"
    )
    assert session_cm.enter_task is session_cm.exit_task, (
        "ClientSession must enter and exit on the same asyncio task "
        f"(entered={session_cm.enter_task!r}, exited={session_cm.exit_task!r})"
    )

    # Sanity: that single task is neither the connect() nor the disconnect()
    # caller. The earlier broken version entered in connect_task and exited in
    # disconnect_task; the supervisor design owns both.
    assert transport_cm.enter_task is not connect_task
    assert transport_cm.enter_task is not disconnect_task


async def test_double_connect_warns_and_does_not_raise(monkeypatch, caplog):
    """A second ``connect()`` while already connected must not raise; it must
    log a warning and leave the existing connection untouched."""
    _install_transport_mocks(monkeypatch)
    server = MCPServerRemote(url="http://test.example/mcp")
    await server.connect()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        await server.connect()

    assert "Already connected" in caplog.text

    async def cleanup() -> None:
        await server.disconnect()

    await asyncio.create_task(cleanup())


async def test_double_connect_does_not_reopen_transports(monkeypatch):
    """The early-return on ``_is_connected`` must short-circuit before touching
    the transport stack again."""
    transport_cm, session_cm = _install_transport_mocks(monkeypatch)
    server = MCPServerRemote(url="http://test.example/mcp")

    await server.connect()
    assert transport_cm.enter_count == 1
    assert session_cm.enter_count == 1

    await server.connect()  # second call — must noop
    assert transport_cm.enter_count == 1
    assert session_cm.enter_count == 1

    async def cleanup() -> None:
        await server.disconnect()

    await asyncio.create_task(cleanup())
