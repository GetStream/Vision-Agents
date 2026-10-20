import json
import os
from typing import AsyncIterator

import av
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from dotenv import load_dotenv
from google.genai import Client, types
from google.genai.client import AsyncClient
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import SEVERAL_SPEAKERS
from vision_agents.plugins.gemini import VLM

from vision_agents.testing import answer, collect_simple_response

load_dotenv()

ALICE = Participant(original=None, user_id="alice", id="alice-1", name="Alice")
BOB = Participant(original=None, user_id="bob", id="bob-1", name="Bob")


@pytest.fixture
async def gemini_api() -> AsyncIterator[tuple[AsyncClient, list[dict]]]:
    """A real server standing in for Gemini: a client of it, and every request body sent.

    Every answer is one streamed "Ok": the tests are about requests.
    """
    bodies: list[dict] = []
    ok = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Ok"}]}}]}

    async def stream(request: web.Request) -> web.StreamResponse:
        bodies.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(f"data: {json.dumps(ok)}\n\n".encode())
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1beta/models/{call}", stream)
    server = TestServer(app)
    await server.start_server()
    http_options = types.HttpOptions(base_url=str(server.make_url("")))
    yield Client(api_key="test", http_options=http_options).aio, bodies
    await server.close()


def _solid_color_frame() -> av.VideoFrame:
    frame_array = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_array[:, :] = [255, 0, 0]
    return av.VideoFrame.from_ndarray(frame_array, format="rgb24")


class TestGeminiVLMRequests:
    async def test_the_model_is_told_once_a_second_person_has_spoken(
        self, gemini_api: tuple[AsyncClient, list[dict]]
    ):
        client, bodies = gemini_api
        conversation = InMemoryConversation("", [])
        vlm = VLM(model="test", client=client)
        vlm.set_conversation(conversation)
        vlm.set_instructions("be brief")

        for participant, text in ((ALICE, "Hi"), (ALICE, "Hello?"), (BOB, "Me too")):
            await answer(vlm, conversation, participant, text)
        await vlm.close()

        told = [body["systemInstruction"]["parts"][0]["text"] for body in bodies]
        # One person: what the base sha (9653cb696) sent; the chat keeps the instructions.
        assert told == ["be brief", "be brief", "be brief\n\n" + SEVERAL_SPEAKERS]
        assert bodies[-1]["contents"][-1]["parts"] == [{"text": "Bob: Me too"}]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
class TestGeminiVLM:
    @pytest.fixture
    async def vlm(self):
        vlm_instance = VLM(model="gemini-3-flash-preview")
        vlm_instance.set_conversation(InMemoryConversation("be brief", []))
        yield vlm_instance
        await vlm_instance.close()

    async def test_gemini_vlm_simple_response(self, vlm: VLM):
        vlm.add_frame(_solid_color_frame())

        deltas, final = await collect_simple_response(
            vlm.simple_response("Describe the scene.")
        )
        await vlm.events.wait()

        assert deltas
        assert final.text
        assert vlm.metrics.agent_metrics.vlm_inferences__total.value() == 1
