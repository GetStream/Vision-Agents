import logging

import pytest
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
from vision_agents.core.vad.silero import SileroVADSessionPool
from vision_agents.plugins.smart_turn.smart_turn_detection import SmartTurnDetection

logger = logging.getLogger(__name__)


@pytest.fixture
async def smart_turn():
    td = SmartTurnDetection()
    await td.warmup()
    await td.start()
    yield td
    await td.stop()


@pytest.mark.skip_blockbuster
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
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        event_order = []

        # Subscribe to events
        @smart_turn.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @smart_turn.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        for pcm in mia_audio_16khz.chunks(chunk_size=304):
            await smart_turn.process_audio(pcm, participant, conversation)

        # Wait for background processing to complete
        await smart_turn.wait_for_processing_complete()

        assert event_order == ["start", "stop"] or event_order == [
            "start",
            "stop",
            "start",
            "stop",
        ]

    async def test_turn_detection(self, smart_turn, mia_audio_16khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])
        event_order = []

        # Subscribe to events
        @smart_turn.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @smart_turn.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        await smart_turn.process_audio(mia_audio_16khz, participant, conversation)

        # Wait for background processing to complete
        await smart_turn.wait_for_processing_complete()

        # Verify that turn detection is working - we should get at least some turn events
        # With continuous processing, we may get multiple start/stop cycles
        assert event_order == ["start", "stop"] or event_order == [
            "start",
            "stop",
            "start",
            "stop",
        ]

    """
    TODO
    - Test that the 2nd turn detect includes the audio from the first turn
    - Test that turn detection is ran after 8s of audio
    - Test that turn detection is run after speech and 2s of silence
    - Test that silence doens't start a new segmetn
    - Test that speaking starts a new segment

    """
