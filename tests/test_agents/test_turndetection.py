import asyncio
from typing import Any, Optional

from getstream.video.rtc import AudioStreamTrack
from vision_agents.core import Agent, User
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge import Call, EdgeTransport
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm.events import LLMResponseCompletedEvent
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.stt import STT
from vision_agents.core.stt.events import STTPartialTranscriptEvent, STTTranscriptEvent
from vision_agents.core.turn_detection import TurnEndedEvent


class DummySTT(STT):
    """Minimal STT that registers STT/turn events but does no processing."""

    async def process_audio(self, *_: object, **__: object) -> None:
        pass


class DummyEdge(EdgeTransport):
    def __init__(self):
        super(DummyEdge, self).__init__()
        self.events = EventManager()

    async def authenticate(self, user: User) -> None: ...

    async def create_call(
        self, call_id: str, agent_user_id: Optional[str] = None, **kwargs
    ) -> Call: ...

    def create_audio_track(self, *args, **kwargs) -> AudioStreamTrack:
        return AudioStreamTrack(
            audio_buffer_size_ms=300_000,
            sample_rate=48000,
            channels=2,
        )

    async def close(self):
        pass

    def open_demo(self, *args, **kwargs):
        pass

    async def join(self, *args, **kwargs): ...
    async def publish_tracks(self, audio_track, video_track): ...

    async def create_conversation(self, call: Any, user: User, instructions): ...

    def add_track_subscriber(self, track_id: str): ...

    async def send_custom_event(self, data: dict) -> None: ...


class EventEmittingLLM(LLM):
    """LLM that emits LLMResponseCompletedEvent on its event bus, like real implementations."""

    async def simple_response(
        self, text: str, participant: Optional[Participant] = None
    ) -> LLMResponseEvent[Any]:
        self.events.send(
            LLMResponseCompletedEvent(
                plugin_name="test",
                original=None,
                text="LLM response",
                item_id="llm-item-1",
            )
        )
        return LLMResponseEvent(text="LLM response", original=None)


class TestEagerTurnDetection:
    """Tests for eager turn detection and conversation ordering."""

    async def test_eager_turn_conversation_ordering(self, participant):
        """Eager turn detection must not record assistant message before user message."""
        llm = EventEmittingLLM()
        agent = Agent(
            llm=llm,
            stt=DummySTT(),
            edge=DummyEdge(),
            agent_user=User(id="agent-1", name="bot"),
        )
        agent.conversation = InMemoryConversation(instructions="", messages=[])

        # 1. Partial transcript arrives (simulates Deepgram EagerEndOfTurn)
        agent.events.send(
            STTPartialTranscriptEvent(text="Hello there", participant=participant)
        )
        await agent.events.wait()

        # 2. Eager turn end → starts LLM speculatively
        agent.events.send(
            TurnEndedEvent(
                participant=participant, eager_end_of_turn=True, confidence=0.9
            )
        )
        await agent.events.wait()
        # Let the LLM task (created via asyncio.create_task) complete
        await asyncio.sleep(0.1)
        await agent.events.wait()

        # 3. Final transcript arrives → should write user message to conversation
        agent.events.send(
            STTTranscriptEvent(text="Hello there", participant=participant)
        )
        await agent.events.wait()

        # 4. Final turn end → confirms turn, should trigger _finish_llm_turn
        agent.events.send(
            TurnEndedEvent(
                participant=participant, eager_end_of_turn=False, confidence=0.95
            )
        )
        await agent.events.wait()

        # Assert: user message must appear before assistant message
        roles = [m.role for m in agent.conversation.messages]
        assert "user" in roles, f"Expected user message in conversation. Got: {roles}"
        assert "assistant" in roles, (
            f"Expected assistant message in conversation. Got: {roles}"
        )
        assert roles.index("user") < roles.index("assistant"), (
            f"User message should come before assistant. Got: {roles}"
        )
