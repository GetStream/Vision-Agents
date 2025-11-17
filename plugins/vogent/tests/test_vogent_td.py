"""
Basic tests for Vogent turn detection.
"""

import asyncio

import pytest

from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
import logging

from vision_agents.plugins.vogent.vogent_turn_detection import VogentTurnDetection

logger = logging.getLogger(__name__)


class TestVogentTurn:
    @pytest.fixture
    async def td(self):
        td = VogentTurnDetection()
        try:
            await td.start()
            yield td
        finally:
            await td.stop()

    @pytest.mark.skip(reason="This test is temporarily disabled.")
    async def test_turn_detection(self, td, mia_audio_16khz, silence_2s_48khz):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])
        event_order = []

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Vogent turn started on {event.session_id}")
            event_order.append("start")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Vogent turn ended on {event.session_id}")
            event_order.append("stop")

        await td.process_audio(mia_audio_16khz, participant, conversation)
        await td.process_audio(silence_2s_48khz, participant, conversation)

        await asyncio.sleep(0.001)

        await asyncio.sleep(5)

        # Verify that turn detection is working - we should get at least some turn events
        assert event_order == ["start", "stop"] or event_order == [
            "start",
            "stop",
            "start",
            "stop",
        ]
