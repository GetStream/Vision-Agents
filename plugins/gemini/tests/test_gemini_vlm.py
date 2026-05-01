import os

import av
import numpy as np
import pytest
from dotenv import load_dotenv
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.plugins.gemini import VLM

from tests.utils import collect_simple_response

load_dotenv()


def _solid_color_frame() -> av.VideoFrame:
    frame_array = np.zeros((64, 64, 3), dtype=np.uint8)
    frame_array[:, :] = [255, 0, 0]
    return av.VideoFrame.from_ndarray(frame_array, format="rgb24")


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
