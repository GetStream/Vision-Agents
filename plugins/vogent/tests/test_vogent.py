import logging

import pytest
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
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
        await td.close()


@pytest.mark.skip()
@pytest.mark.skip_blockbuster
@pytest.mark.integration
class TestVogentTurnDetection:
    async def test_turn_detection(
        self, vogent_turn_detection, mia_audio_16khz, silence_2s_48khz
    ):
        participant = Participant(user_id="mia", original={}, id="mia")
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        await vogent_turn_detection.process_audio(
            mia_audio_16khz, participant, conversation
        )
        await vogent_turn_detection.process_audio(
            silence_2s_48khz, participant, conversation
        )

        await vogent_turn_detection.wait_for_processing_complete()

        items = await vogent_turn_detection.output.collect(timeout=1.0)
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
