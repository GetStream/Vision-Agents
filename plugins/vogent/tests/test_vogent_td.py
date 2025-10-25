"""
Basic tests for Vogent turn detection.
"""
import asyncio

import pytest

from plugins.deepgram.vision_agents.plugins import deepgram
from plugins.vogent.vision_agents.plugins.vogent import TurnDetection
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent
from vision_agents.core.turn_detection import TurnEndedEvent, TurnStartedEvent
import logging

logger = logging.getLogger(__name__)

class TestVogentTurn:

    @pytest.fixture
    async def td(self):
        td = TurnDetection()
        try:
            td.start()
            yield td
        finally:
            td.stop()

    async def test_turn_detection(self, td, mia_audio_48khz_chunked):
        participant = Participant(user_id="mia", original={})
        conversation = InMemoryConversation(instructions="be nice", messages=[])

        stt = deepgram.STT()
        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            logger.info(f"Transcript, {event.text}")

        @stt.events.subscribe
        async def on_partial_transcript(event: STTPartialTranscriptEvent):
            logger.info(f"P Transcript, {event.text}")

        # Subscribe to events
        @td.events.subscribe
        async def on_start(event: TurnStartedEvent):
            logger.info(f"Vogent turn started on {event.session_id}")

        @td.events.subscribe
        async def on_stop(event: TurnEndedEvent):
            logger.info(f"Vogent turn ended on {event.session_id}")

        # Process each 20ms audio chunk
        for chunk in mia_audio_48khz_chunked:
            logger.info("Chunk: .")
            await stt.process_audio(chunk, participant)
            await td.process_audio(chunk, participant, conversation)

        await asyncio.sleep(2)


