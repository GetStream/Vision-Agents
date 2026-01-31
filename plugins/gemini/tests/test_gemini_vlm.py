import os

import av
import numpy as np
import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseCompletedEvent,
    VLMInferenceCompletedEvent,
)
from vision_agents.plugins.gemini import VLM

load_dotenv()


def _solid_color_frame() -> av.VideoFrame:
    frame_array = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_array[:, :] = [255, 0, 0]
    return av.VideoFrame.from_ndarray(frame_array, format="rgb24")


@pytest.fixture
async def vlm() -> VLM:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY not set")

    vlm_instance = VLM(model="gemini-3-flash-preview", api_key=api_key)
    vlm_instance.set_conversation(InMemoryConversation("be brief", []))
    try:
        yield vlm_instance
    finally:
        await vlm_instance.close()


@pytest.mark.integration
async def test_gemini_vlm_simple_response(vlm: VLM):
    vlm._frame_buffer.append(_solid_color_frame())

    events: list[LLMResponseCompletedEvent | VLMInferenceCompletedEvent] = []

    @vlm.events.subscribe
    async def handle_event(
        event: LLMResponseCompletedEvent | VLMInferenceCompletedEvent,
    ):
        events.append(event)

    response = await vlm.simple_response("Describe the scene.")
    await vlm.events.wait()

    assert response.text
    assert any(e.type == "plugin.vlm_inference_completed" for e in events)
    assert any(e.type == "plugin.llm_response_completed" for e in events)
