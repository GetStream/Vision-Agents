import asyncio

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.stt import STT as BaseSTT
from vision_agents.core.turn_detection import TurnEndedEvent


class DummySTT(BaseSTT):
    async def process_audio(self, pcm_data, participant):
        pass


class TestSTTTurnDuration:
    async def test_turn_ended_event_includes_duration_after_turn_started(self):
        stt = DummySTT()
        participant = Participant({}, user_id="u1", id="p1")
        received: list[TurnEndedEvent] = []
        delivered = asyncio.Event()

        @stt.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            received.append(event)
            delivered.set()

        stt._emit_turn_started_event(participant=participant)
        stt._emit_turn_ended_event(participant=participant)

        await asyncio.wait_for(delivered.wait(), timeout=2.0)

        assert len(received) == 1
        assert received[0].duration_ms is not None
        assert received[0].duration_ms > 0

    async def test_turn_ended_without_turn_started_has_no_duration(self):
        stt = DummySTT()
        participant = Participant({}, user_id="u1", id="p1")
        received: list[TurnEndedEvent] = []
        delivered = asyncio.Event()

        @stt.events.subscribe
        async def on_turn_ended(event: TurnEndedEvent):
            received.append(event)
            delivered.set()

        stt._emit_turn_ended_event(participant=participant)

        await asyncio.wait_for(delivered.wait(), timeout=2.0)

        assert len(received) == 1
        assert received[0].duration_ms is None
