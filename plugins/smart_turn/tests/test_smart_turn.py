import logging

import pytest
from conftest import skip_if_huggingface_model_unavailable
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.core.vad.silero import SileroVADSessionPool
from vision_agents.plugins.smart_turn.smart_turn_detection import SmartTurnDetection

logger = logging.getLogger(__name__)


@pytest.fixture
async def smart_turn():
    td = SmartTurnDetection()
    try:
        await td.warmup()
    except Exception as exc:
        skip_if_huggingface_model_unavailable(exc, "Smart Turn model")
        raise
    await td.start()
    yield td
    await td.close()


class TestSmartTurn:
    async def test_silero_predict(self, mia_audio_16khz, tmp_path):
        vad_pool = await SileroVADSessionPool.load(tmp_path.as_posix())
        vad = vad_pool.session()

        for pcm_chunk in mia_audio_16khz.chunks(chunk_size=512):
            if len(pcm_chunk.samples) != 512:
                continue
            result = vad.predict_speech(pcm_chunk)
            assert 1.0 > result > 0.0

    async def test_turn_detection_chunks(self, smart_turn, mia_audio_16khz):
        participant = Participant(user_id="mia", id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        for pcm in mia_audio_16khz.chunks(chunk_size=304):
            await smart_turn.process_audio(pcm, participant, conversation)

        await smart_turn.wait_for_processing_complete()

        items = await smart_turn.output.collect(timeout=1.0)
        kinds = [
            "start"
            if isinstance(item, TurnStarted)
            else "stop"
            if isinstance(item, TurnEnded)
            else None
            for item in items
        ]
        assert kinds == ["start", "stop"] or kinds == [
            "start",
            "stop",
            "start",
            "stop",
        ]

    async def test_turn_detection(self, smart_turn, mia_audio_16khz):
        participant = Participant(user_id="mia", id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        await smart_turn.process_audio(mia_audio_16khz, participant, conversation)

        await smart_turn.wait_for_processing_complete()

        items = await smart_turn.output.collect(timeout=1.0)
        kinds = [
            "start"
            if isinstance(item, TurnStarted)
            else "stop"
            if isinstance(item, TurnEnded)
            else None
            for item in items
        ]
        # With continuous processing, we may get multiple start/stop cycles
        assert kinds == ["start", "stop"] or kinds == [
            "start",
            "stop",
            "start",
            "stop",
        ]

    async def test_silence_does_not_start_segment(self, smart_turn, silence_1s_16khz):
        participant = Participant(user_id="mia", id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        await smart_turn.process_audio(silence_1s_16khz, participant, conversation)
        await smart_turn.wait_for_processing_complete()

        items = await smart_turn.output.collect(timeout=0.5)
        assert items == []

    async def test_speech_starts_segment(self, smart_turn, mia_audio_16khz):
        participant = Participant(user_id="mia", id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        await smart_turn.process_audio(mia_audio_16khz, participant, conversation)
        await smart_turn.wait_for_processing_complete()

        items = await smart_turn.output.collect(timeout=1.0)
        assert any(isinstance(item, TurnStarted) for item in items)

    """
    TODO
    - Test that the 2nd turn detect includes the audio from the first turn
    - Test that turn detection is ran after 8s of audio
    - Test that turn detection is run after speech and 2s of silence
    """
