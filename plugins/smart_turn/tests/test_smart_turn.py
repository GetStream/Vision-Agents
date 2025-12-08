import tempfile

import pytest

from conftest import skip_blockbuster
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnStartedEvent, TurnEndedEvent
from vision_agents.core.vad.silero import prepare_silero_vad
from vision_agents.plugins.smart_turn.smart_turn_detection import SmartTurnDetection

import logging

logger = logging.getLogger(__name__)


@skip_blockbuster
class TestSmartTurn:
    @pytest.fixture
    async def td(self):
        td = SmartTurnDetection()
        await td.start()
        yield td

    async def test_silero_predict(self, mia_audio_16khz):
        vad = await prepare_silero_vad(tempfile.gettempdir())

        for pcm_chunk in mia_audio_16khz.chunks(chunk_size=512):
            if len(pcm_chunk.samples) != 512:
                continue
            result = vad.predict_speech(pcm_chunk)
            assert 1.0 > result > 0.0

    async def test_turn_detection_chunks(self, td, mia_audio_16khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        event_order = []

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        for pcm in mia_audio_16khz.chunks(chunk_size=304):
            await td.process_audio(pcm, participant, conversation)

        # Wait for background processing to complete
        await td.wait_for_processing_complete()

        assert event_order == ["start", "stop"] or event_order == [
            "start",
            "stop",
            "start",
            "stop",
        ]

    async def test_turn_detection(self, td, mia_audio_16khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])
        event_order = []

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Smart turn turn started on {event.session_id}")
            event_order.append("start")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Smart turn turn ended on {event.session_id}")
            event_order.append("stop")

        await td.process_audio(mia_audio_16khz, participant, conversation)

        # Wait for background processing to complete
        await td.wait_for_processing_complete()

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
