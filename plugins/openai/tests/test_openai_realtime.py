from unittest.mock import AsyncMock, MagicMock

import pytest
import asyncio

from getstream.plugins import OpenAIRealtime
from getstream.video.call import Call
from getstream.video.openai import ConnectionManagerWrapper


@pytest.fixture
def mock_call():
    call = MagicMock(spec=Call)
    call.call_type = "default"
    call.id = "test-call"
    return call


@pytest.fixture
def mock_connection():
    conn = MagicMock(spec=ConnectionManagerWrapper)
    # prepare nested mocks used by methods
    conn.session = AsyncMock()
    conn.conversation = AsyncMock()
    conn.conversation.item = AsyncMock()
    conn.response = AsyncMock()
    # Add context manager support
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock()
    return conn


@pytest.mark.asyncio
async def test_connect_updates_state_and_emits_event(mock_call, mock_connection):
    # call.connect_openai should return our mocked connection
    mock_call.connect_openai.return_value = mock_connection  # type: ignore[attr-defined]

    sts = OpenAIRealtime(api_key="key123", voice="alloy", instructions="hi")
    events = []

    @sts.on("connected")  # type: ignore[arg-type]
    async def _on_connected(event):
        events.append("connected")

    async with await sts.connect(mock_call, agent_user_id="assistant") as connection:
        await asyncio.sleep(0)  # allow event callbacks to run

    assert sts.is_connected is True
    assert connection is mock_connection
    assert "connected" in events

    # Ensure call.connect_openai invoked with the right params
    mock_call.connect_openai.assert_called_once_with(
        "key123", "assistant", model="gpt-4o-realtime-preview"
    )


@pytest.mark.asyncio
async def test_update_session_calls_underlying_client(mock_call, mock_connection):
    mock_call.connect_openai.return_value = mock_connection  # type: ignore[attr-defined]
    sts = OpenAIRealtime(api_key="abc")
    await sts.connect(mock_call)

    await sts.update_session(voice="nova", temperature=0.5)
    mock_connection.session.update.assert_awaited_once_with(
        session={"voice": "nova", "temperature": 0.5}
    )


@pytest.mark.asyncio
async def test_send_user_message(mock_call, mock_connection):
    mock_call.connect_openai.return_value = mock_connection  # type: ignore[attr-defined]
    sts = OpenAIRealtime(api_key="abc")
    await sts.connect(mock_call)

    await sts.send_user_message("hello")

    mock_connection.conversation.item.create.assert_awaited()
    args, kwargs = mock_connection.conversation.item.create.await_args
    # Basic structure check
    assert kwargs["item"]["role"] == "user"
    assert kwargs["item"]["content"][0]["text"] == "hello"

    mock_connection.response.create.assert_awaited()


@pytest.mark.asyncio
async def test_send_function_call_output(mock_call, mock_connection):
    mock_call.connect_openai.return_value = mock_connection  # type: ignore[attr-defined]
    sts = OpenAIRealtime(api_key="xyz")
    await sts.connect(mock_call)

    await sts.send_function_call_output("tool-123", '{"ok": true}')

    mock_connection.conversation.item.create.assert_awaited_once_with(
        item={
            "type": "function_call_output",
            "call_id": "tool-123",
            "output": '{"ok": true}',
        }
    )


@pytest.mark.asyncio
async def test_request_assistant_response(mock_call, mock_connection):
    mock_call.connect_openai.return_value = mock_connection  # type: ignore[attr-defined]
    sts = OpenAIRealtime(api_key="abc")
    await sts.connect(mock_call)

    await sts.request_assistant_response()

    mock_connection.response.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_methods_raise_if_not_connected():
    sts = OpenAIRealtime(api_key="abc")
    with pytest.raises(RuntimeError):
        await sts.update_session(foo="bar")
    with pytest.raises(RuntimeError):
        await sts.send_user_message("hi")
    with pytest.raises(RuntimeError):
        await sts.send_function_call_output("id", "out")
    with pytest.raises(RuntimeError):
        await sts.request_assistant_response()
