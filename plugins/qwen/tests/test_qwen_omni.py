import os

import dotenv
import pytest
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.plugins.qwen import QwenOmni

from tests.base_test import BaseTest

dotenv.load_dotenv()


@pytest.fixture()
async def conversation():
    return InMemoryConversation("", [])


@pytest.fixture()
async def qwen_omni(conversation):
    if not os.getenv("ALIBABA_API_KEY"):
        pytest.skip("ALIBABA_API_KEY not set; skipping integration test")

    llm_ = QwenOmni(model="qwen3-omni-flash")
    llm_.set_conversation(conversation)
    return llm_


class TestQwenOmni(BaseTest):
    @pytest.mark.integration
    async def test_simple_response_text_only(self, qwen_omni, conversation):
        """Test simple text response with real API."""
        await conversation.send_message(role="user", user_id="id1", content="Say hello")
        response = await qwen_omni.simple_response(text="Say hello")

        assert response.text is not None
        assert len(response.text) > 0

    @pytest.mark.integration
    async def test_simple_response_streaming_text(self, qwen_omni, conversation):
        """Test streaming text responses with real API."""
        events = []

        @qwen_omni.events.subscribe
        async def listen(
            event: LLMResponseChunkEvent | LLMResponseCompletedEvent,
        ):
            events.append(event)

        await conversation.send_message(
            role="user", user_id="id1", content="Count to three"
        )
        response = await qwen_omni.simple_response(text="Count to three")
        await qwen_omni.events.wait(0.5)

        assert response.text is not None
        assert len(response.text) > 0

        chunk_events = [e for e in events if e.type == "plugin.llm_response_chunk"]
        completed_events = [
            e for e in events if e.type == "plugin.llm_response_completed"
        ]

        assert len(chunk_events) > 0
        assert len(completed_events) == 1
        assert completed_events[0].text == response.text

    @pytest.mark.integration
    async def test_simple_response_audio_output(self, qwen_omni, conversation):
        """Test audio output from real API response."""
        events = []

        @qwen_omni.events.subscribe
        async def listen(event: RealtimeAudioOutputEvent):
            events.append(event)

        await conversation.send_message(role="user", user_id="id1", content="Say hello")
        await qwen_omni.simple_response(text="Say hello")
        await qwen_omni.events.wait(1.0)

        audio_events = [e for e in events if e.type == "plugin.realtime_audio_output"]
        assert len(audio_events) > 0
        assert audio_events[0].data is not None
        assert audio_events[0].data.sample_rate == 24000

    @pytest.mark.integration
    async def test_close_cleanup(self, qwen_omni):
        """Test that close cleans up resources."""
        await qwen_omni.close()

        assert qwen_omni._video_forwarder is None

    @pytest.mark.integration
    async def test_simple_response_no_conversation(self):
        """Test that simple_response handles missing conversation gracefully."""
        if not os.getenv("ALIBABA_API_KEY"):
            pytest.skip("ALIBABA_API_KEY not set; skipping integration test")

        qwen_omni = QwenOmni(model="qwen3-omni-flash")
        qwen_omni._conversation = None

        response = await qwen_omni.simple_response(text="test")

        assert response.text == ""
        await qwen_omni.close()

    @pytest.mark.integration
    async def test_inherits_from_video_llm(self, qwen_omni):
        """Test that QwenOmni inherits from VideoLLM, not AudioLLM."""
        from vision_agents.core.llm.llm import VideoLLM, AudioLLM

        assert isinstance(qwen_omni, VideoLLM)
        assert not isinstance(qwen_omni, AudioLLM)
