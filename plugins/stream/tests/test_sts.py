import asyncio
import json
from typing import Any, AsyncIterator, Optional

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from vision_agents.core.llm.tool_call import CallNotHeld, ToolCall, current_call
from vision_agents.plugins import stream

SETTLE = 2.0


class Conversation:
    """A stand-in for the router's speech-to-speech socket, which answers start with started."""

    def __init__(self):
        self.frames: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.url = ""
        self._socket: Optional[web.WebSocketResponse] = None

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/sts/stream", self._stream)
        return app

    async def send(self, frame: dict[str, Any]) -> None:
        assert self._socket is not None
        await self._socket.send_json(frame)

    async def told(self) -> dict[str, Any]:
        """The next frame the client sent, past its start."""
        return await asyncio.wait_for(self.frames.get(), SETTLE)

    async def _stream(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self._socket = socket
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                continue
            frame = json.loads(message.data)
            if frame.get("type") == "start":
                await socket.send_json(
                    {"type": "started", "provider": "stub", "model": "stub-sts"}
                )
                continue
            await self.frames.put(frame)
        return socket


class TestSTSTools:
    @pytest.fixture
    async def conversation(self) -> AsyncIterator[Conversation]:
        fake = Conversation()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    async def model(self, conversation: Conversation) -> AsyncIterator[stream.STS]:
        llm = stream.STS(
            target="stub/stub-sts", url=conversation.url, customer_id="acme"
        )
        yield llm
        await llm.close()

    async def test_a_function_reads_the_call_it_is_running_for(
        self, conversation: Conversation, model: stream.STS
    ):
        seen: list[ToolCall] = []

        @model.register_function(description="Look the caller up")
        async def whoami() -> str:
            seen.append(current_call())
            return "ok"

        await model.connect()
        await conversation.send(
            {"type": "tool_call", "id": "call-1", "name": "whoami", "arguments": "{}"}
        )

        assert (await conversation.told())["output"] == "ok"
        assert seen[0].id == "call-1"

    async def test_a_function_tells_the_caller_how_it_is_going(
        self, conversation: Conversation, model: stream.STS
    ):
        @model.register_function(description="Check the book")
        async def check() -> str:
            await current_call().update("checking the patio")
            await current_call().say("One moment.")
            return "a table at 7:30"

        await model.connect()
        await conversation.send(
            {"type": "tool_call", "id": "call-2", "name": "check", "arguments": "{}"}
        )

        assert await conversation.told() == {
            "type": "tool_update",
            "tool_call_id": "call-2",
            "text": "checking the patio",
            "verbatim": False,
        }
        assert await conversation.told() == {
            "type": "tool_update",
            "tool_call_id": "call-2",
            "text": "One moment.",
            "verbatim": True,
        }
        assert await conversation.told() == {
            "type": "tool_result",
            "tool_call_id": "call-2",
            "output": "a table at 7:30",
        }

    async def test_a_held_call_goes_on_once_the_router_holds_it(
        self, conversation: Conversation, model: stream.STS
    ):
        @model.register_function(description="Charge the card")
        async def charge() -> str:
            await current_call().disallow_interruptions()
            return "charged"

        await model.connect()
        await conversation.send(
            {"type": "tool_call", "id": "call-3", "name": "charge", "arguments": "{}"}
        )

        assert await conversation.told() == {
            "type": "tool_hold",
            "tool_call_id": "call-3",
        }
        await conversation.send({"type": "tool_held", "id": "call-3"})
        assert (await conversation.told())["output"] == "charged"

    async def test_a_call_the_model_cancelled_is_not_held(
        self, conversation: Conversation, model: stream.STS
    ):
        refused: list[BaseException] = []

        @model.register_function(description="Charge the card")
        async def charge() -> str:
            try:
                await current_call().disallow_interruptions()
            except CallNotHeld as error:
                refused.append(error)
                raise
            return "charged"

        await model.connect()
        await conversation.send(
            {"type": "tool_call", "id": "call-4", "name": "charge", "arguments": "{}"}
        )

        assert (await conversation.told())["type"] == "tool_hold"
        await conversation.send({"type": "tool_cancel", "ids": ["call-4"]})
        answer = await conversation.told()

        assert refused, "the function learns its call was not held"
        assert answer["tool_call_id"] == "call-4"
        assert "error" in answer
