from typing import AsyncIterator

import openai
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer


@pytest.fixture
async def openai_api() -> AsyncIterator[
    tuple[openai.AsyncOpenAI, list[dict[str, object]]]
]:
    """A real server standing in for OpenAI: a client of it, and every request body sent.

    Every answer is an empty stream: the tests are about requests.
    """
    bodies: list[dict[str, object]] = []

    async def conversation(request: web.Request) -> web.Response:
        return web.json_response(
            {"id": "conv_1", "object": "conversation", "created_at": 0, "metadata": {}}
        )

    async def stream(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/conversations", conversation)
    app.router.add_post("/v1/responses", stream)
    app.router.add_post("/v1/chat/completions", stream)
    server = TestServer(app)
    await server.start_server()
    client = openai.AsyncOpenAI(
        api_key="test", base_url=str(server.make_url("/v1")), max_retries=0
    )
    yield client, bodies
    await client.close()
    await server.close()
