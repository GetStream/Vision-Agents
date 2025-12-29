import asyncio
import logging

import pytest
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
from vision_agents.plugins.vogent.vogent_turn_detection import VogentTurnDetection

logger = logging.getLogger(__name__)


@pytest.fixture
async def vogent_turn_detection():
    td = VogentTurnDetection()
    await td.warmup()
    await td.start()
    try:
        yield td
    finally:
        await td.stop()


@pytest.mark.skip_blockbuster
@pytest.mark.integration
class TestVogentTurnDetection:
    async def test_turn_detection(
        self, vogent_turn_detection, mia_audio_16khz, silence_2s_48khz
    ):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])
        event_order = []

        # Subscribe to events
        @vogent_turn_detection.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Vogent turn started on {event.session_id}")
            event_order.append("start")

        @vogent_turn_detection.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Vogent turn ended on {event.session_id}")
            event_order.append("stop")

        await vogent_turn_detection.process_audio(
            mia_audio_16khz, participant, conversation
        )
        await vogent_turn_detection.process_audio(
            silence_2s_48khz, participant, conversation
        )

        await asyncio.sleep(0.001)

        await asyncio.sleep(5)

        # Verify that turn detection is working - we should get at least some turn events
        assert event_order == ["start", "stop"] or event_order == [
            "start",
            "stop",
            "start",
            "stop",
        ]
