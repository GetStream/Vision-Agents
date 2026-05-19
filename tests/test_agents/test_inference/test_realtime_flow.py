import asyncio
from typing import AsyncIterator

import numpy as np
import pytest
from getstream.video.rtc import PcmData
from getstream.video.rtc.track_util import AudioFormat
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.agents.events import (
    AgentTurnEndedEvent,
    AgentTurnStartedEvent,
    UserTurnEndedEvent,
    UserTurnStartedEvent,
)
from vision_agents.core.agents.inference.audio import (
    AudioInputChunk,
    AudioInputStream,
    AudioOutputChunk,
    AudioOutputFlush,
    AudioOutputStream,
)
from vision_agents.core.agents.inference.realtime_flow import (
    RealtimeInferenceFlow,
)
from vision_agents.core.agents.transcript import TranscriptStore
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm.realtime import (
    Realtime,
    RealtimeAgentSpeechEnded,
    RealtimeAgentSpeechStarted,
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeAudioOutputDone,
    RealtimeUserSpeechEnded,
    RealtimeUserSpeechStarted,
    RealtimeUserTranscript,
)
from vision_agents.core.utils.stream import Stream

from .stubs import RealtimeStub


def _silence(samples: int = 320) -> PcmData:
    return PcmData(
        samples=np.zeros(samples, dtype=np.int16),
        sample_rate=16000,
        format=AudioFormat.S16,
    )


@pytest.fixture
def participant() -> Participant:
    return Participant(id="p1", user_id="u1", original=None)


@pytest.fixture
def transcripts() -> TranscriptStore:
    return TranscriptStore(agent_user_id="agent-1")


@pytest.fixture
def conversation() -> InMemoryConversation:
    return InMemoryConversation(instructions="", messages=[])


@pytest.fixture
async def events() -> EventManager:
    return EventManager()


@pytest.fixture
def flow_factory(transcripts, conversation, events):
    def _build(
        *,
        llm: Realtime | None = None,
        audio_input: AudioInputStream | None = None,
        audio_output: AudioOutputStream | None = None,
    ) -> RealtimeInferenceFlow:
        return RealtimeInferenceFlow(
            audio_input=audio_input or AudioInputStream(),
            audio_output=audio_output or AudioOutputStream(),
            llm=llm or RealtimeStub(),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
        )

    return _build


class TestProcessAudioInput:
    async def test_audio_chunks_drive_llm_to_emit_transcripts(
        self, flow_factory, participant, conversation
    ) -> None:
        index = 0

        async def echo(
            pcm: PcmData, p: Participant
        ) -> AsyncIterator[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
        ]:
            nonlocal index
            index += 1
            yield RealtimeAgentTranscript(text=f"chunk-{index}", mode="final")

        llm = RealtimeStub.from_callable(audio_factory=echo)
        audio_input = AudioInputStream()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_input=audio_input, audio_output=audio_output)

        in_task = asyncio.create_task(flow.process_audio_input(audio_input))
        out_task = asyncio.create_task(
            flow.process_llm_output(llm.output, audio_output)
        )

        await audio_input.send(
            AudioInputChunk(data=_silence(), participant=participant)
        )
        await audio_input.send(
            AudioInputChunk(data=_silence(), participant=participant)
        )

        audio_input.close()
        await in_task
        llm.output.close()
        await out_task

        assistant_msgs = [m for m in conversation.messages if m.role == "assistant"]
        assert [m.content for m in assistant_msgs] == ["chunk-1", "chunk-2"]

    async def test_continues_after_llm_failure_on_one_chunk(
        self, flow_factory, participant, conversation
    ) -> None:
        first = True

        async def flaky(
            pcm: PcmData, p: Participant
        ) -> AsyncIterator[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
        ]:
            nonlocal first
            if first:
                first = False
                raise RuntimeError("boom")
            yield RealtimeAgentTranscript(text="ok", mode="final")

        llm = RealtimeStub.from_callable(audio_factory=flaky)
        audio_input = AudioInputStream()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_input=audio_input, audio_output=audio_output)

        in_task = asyncio.create_task(flow.process_audio_input(audio_input))
        out_task = asyncio.create_task(
            flow.process_llm_output(llm.output, audio_output)
        )

        await audio_input.send(
            AudioInputChunk(data=_silence(), participant=participant)
        )
        await audio_input.send(
            AudioInputChunk(data=_silence(), participant=participant)
        )

        audio_input.close()
        await in_task
        llm.output.close()
        await out_task

        assistant_msgs = [m for m in conversation.messages if m.role == "assistant"]
        assert [m.content for m in assistant_msgs] == ["ok"]


class TestProcessLLMOutput:
    async def test_audio_output_event_is_forwarded_as_chunk(self, flow_factory) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        # 320 samples @ 16kHz = exactly one 20ms chunk.
        llm.output.send_nowait(RealtimeAudioOutput(data=_silence(320)))
        llm.output.close()
        await task

        items = audio_output.peek()
        assert len(items) == 1
        assert isinstance(items[0], AudioOutputChunk)
        assert items[0].data is not None
        assert items[0].final is False

    async def test_audio_output_done_emits_final_sentinel(self, flow_factory) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(RealtimeAudioOutputDone(interrupted=False))
        llm.output.close()
        await task

        items = audio_output.peek()
        assert len(items) == 1
        assert isinstance(items[0], AudioOutputChunk)
        assert items[0].final is True

    async def test_audio_output_done_interrupted_flushes_pipeline(
        self, flow_factory, transcripts, participant
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)
        transcripts.update_user_transcript(
            participant_id=participant.id,
            user_id=participant.user_id,
            text="hel",
            mode="replacement",
        )
        transcripts.update_agent_transcript(text="hi", mode="replacement")
        epoch_before = llm.epoch

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(RealtimeAudioOutputDone(interrupted=True))
        llm.output.close()
        await task

        assert audio_output.peek() == [AudioOutputFlush()]
        assert (
            transcripts.get_buffer(
                participant_id=participant.id, user_id=participant.user_id
            )
            is None
        )
        assert llm.epoch == epoch_before + 1

    async def test_user_transcript_delta_creates_partial_user_message(
        self, flow_factory, conversation, participant
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(
            RealtimeUserTranscript(participant=participant, mode="delta", text="hel")
        )
        llm.output.close()
        await task

        user_msgs = [m for m in conversation.messages if m.role == "user"]
        assert [m.content for m in user_msgs] == ["hel"]
        assert user_msgs[0].user_id == participant.user_id

    async def test_user_transcript_final_completes_message_and_clears_buffer(
        self, flow_factory, conversation, participant, transcripts
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(
            RealtimeUserTranscript(participant=participant, mode="final", text="hello")
        )
        llm.output.close()
        await task

        user_msgs = [m for m in conversation.messages if m.role == "user"]
        assert [m.content for m in user_msgs] == ["hello"]
        assert (
            transcripts.get_buffer(
                participant_id=participant.id, user_id=participant.user_id
            )
            is None
        )

    async def test_agent_transcript_delta_creates_partial_assistant_message(
        self, flow_factory, conversation
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(RealtimeAgentTranscript(mode="delta", text="hi"))
        llm.output.close()
        await task

        assistant_msgs = [m for m in conversation.messages if m.role == "assistant"]
        assert [m.content for m in assistant_msgs] == ["hi"]
        assert assistant_msgs[0].user_id == "agent-1"

    async def test_agent_transcript_final_completes_message(
        self, flow_factory, conversation
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(RealtimeAgentTranscript(mode="final", text="hi there"))
        llm.output.close()
        await task

        assistant_msgs = [m for m in conversation.messages if m.role == "assistant"]
        assert [m.content for m in assistant_msgs] == ["hi there"]

    async def test_user_transcript_finalizes_pending_agent_transcript(
        self, flow_factory, conversation, participant
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        # Agent starts speaking but never sends a "final" — then the user
        # starts talking. The flow must close out the agent turn before
        # opening the user turn so the conversation reads as ordered,
        # completed messages.
        llm.output.send_nowait(
            RealtimeAgentTranscript(mode="replacement", text="thinking")
        )
        llm.output.send_nowait(
            RealtimeUserTranscript(participant=participant, mode="final", text="hello")
        )
        llm.output.close()
        await task

        roles = [m.role for m in conversation.messages]
        contents = [m.content for m in conversation.messages]
        assert roles == ["assistant", "user"]
        assert contents == ["thinking", "hello"]

    async def test_agent_transcript_finalizes_pending_user_transcript(
        self, flow_factory, conversation, participant
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        task = asyncio.create_task(flow.process_llm_output(llm.output, audio_output))

        llm.output.send_nowait(
            RealtimeUserTranscript(
                participant=participant, mode="replacement", text="hel"
            )
        )
        llm.output.send_nowait(RealtimeAgentTranscript(mode="final", text="hi"))
        llm.output.close()
        await task

        roles = [m.role for m in conversation.messages]
        contents = [m.content for m in conversation.messages]
        assert roles == ["user", "assistant"]
        assert contents == ["hel", "hi"]

    async def test_user_speech_started_emits_user_turn_started(
        self, flow_factory, participant
    ) -> None:
        llm = RealtimeStub()
        flow = flow_factory(llm=llm)
        seen: list[UserTurnStartedEvent] = []

        @flow.events.subscribe
        async def _on(event: UserTurnStartedEvent):
            seen.append(event)

        task = asyncio.create_task(
            flow.process_llm_output(llm.output, AudioOutputStream())
        )
        llm.output.send_nowait(RealtimeUserSpeechStarted(participant=participant))
        llm.output.close()
        await task
        await flow.events.wait()

        assert len(seen) == 1
        assert seen[0].participant is participant

    async def test_user_speech_ended_emits_user_turn_ended(
        self, flow_factory, participant
    ) -> None:
        llm = RealtimeStub()
        flow = flow_factory(llm=llm)
        seen: list[UserTurnEndedEvent] = []

        @flow.events.subscribe
        async def _on(event: UserTurnEndedEvent):
            seen.append(event)

        task = asyncio.create_task(
            flow.process_llm_output(llm.output, AudioOutputStream())
        )
        llm.output.send_nowait(RealtimeUserSpeechEnded(participant=participant))
        llm.output.close()
        await task
        await flow.events.wait()

        assert len(seen) == 1
        assert seen[0].participant is participant

    async def test_agent_speech_started_emits_agent_turn_started(
        self, flow_factory
    ) -> None:
        llm = RealtimeStub()
        flow = flow_factory(llm=llm)
        seen: list[AgentTurnStartedEvent] = []

        @flow.events.subscribe
        async def _on(event: AgentTurnStartedEvent):
            seen.append(event)

        task = asyncio.create_task(
            flow.process_llm_output(llm.output, AudioOutputStream())
        )
        llm.output.send_nowait(RealtimeAgentSpeechStarted())
        llm.output.close()
        await task
        await flow.events.wait()

        assert len(seen) == 1

    async def test_agent_speech_ended_emits_agent_turn_ended_with_interrupted(
        self, flow_factory
    ) -> None:
        llm = RealtimeStub()
        flow = flow_factory(llm=llm)
        seen: list[AgentTurnEndedEvent] = []

        @flow.events.subscribe
        async def _on(event: AgentTurnEndedEvent):
            seen.append(event)

        task = asyncio.create_task(
            flow.process_llm_output(llm.output, AudioOutputStream())
        )
        llm.output.send_nowait(RealtimeAgentSpeechEnded(interrupted=True))
        llm.output.close()
        await task
        await flow.events.wait()

        assert len(seen) == 1
        assert seen[0].interrupted is True


class TestInterrupt:
    async def test_interrupt_appends_flush_to_audio_output(self, flow_factory) -> None:
        audio_output = AudioOutputStream()
        flow = flow_factory(audio_output=audio_output)

        await flow.interrupt()

        assert audio_output.peek() == [AudioOutputFlush()]

    async def test_interrupt_drops_pre_queued_audio_then_appends_flush(
        self, flow_factory
    ) -> None:
        audio_output = AudioOutputStream()
        flow = flow_factory(audio_output=audio_output)
        audio_output.send_nowait(AudioOutputChunk(data=None, final=False))

        await flow.interrupt()

        assert audio_output.peek() == [AudioOutputFlush()]

    async def test_interrupt_clears_buffered_user_and_agent_transcripts(
        self, flow_factory, transcripts, participant
    ) -> None:
        flow = flow_factory()
        transcripts.update_user_transcript(
            participant_id=participant.id,
            user_id=participant.user_id,
            text="hel",
            mode="replacement",
        )
        transcripts.update_agent_transcript(text="hi", mode="replacement")

        await flow.interrupt()

        assert (
            transcripts.get_buffer(
                participant_id=participant.id, user_id=participant.user_id
            )
            is None
        )
        assert transcripts.flush_agent_transcript() is None

    async def test_interrupt_increments_llm_epoch_and_clears_llm_output(
        self, flow_factory
    ) -> None:
        llm = RealtimeStub()
        flow = flow_factory(llm=llm)
        epoch_before = llm.epoch
        llm.output.send_nowait(RealtimeAudioOutputDone(interrupted=False))

        await flow.interrupt()

        assert llm.epoch == epoch_before + 1
        assert llm.output.peek() == []

    async def test_interrupt_is_idempotent(self, flow_factory) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)

        await flow.interrupt()
        await flow.interrupt()

        assert audio_output.peek() == [AudioOutputFlush()]
        assert llm.epoch == 2


class TestSetConversation:
    async def test_set_conversation_routes_subsequent_transcripts_to_new_store(
        self, flow_factory, conversation, participant
    ) -> None:
        llm = RealtimeStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(llm=llm, audio_output=audio_output)
        new_conversation = InMemoryConversation(instructions="", messages=[])

        # First batch: drive event 1 to completion against the original
        # conversation, then swap and drive event 2 against the replacement.
        stream_1: Stream[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
        ] = Stream()
        task_1 = asyncio.create_task(flow.process_llm_output(stream_1, audio_output))
        stream_1.send_nowait(
            RealtimeUserTranscript(participant=participant, mode="final", text="first")
        )
        stream_1.close()
        await task_1

        flow.set_conversation(new_conversation)

        stream_2: Stream[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
        ] = Stream()
        task_2 = asyncio.create_task(flow.process_llm_output(stream_2, audio_output))
        stream_2.send_nowait(
            RealtimeUserTranscript(participant=participant, mode="final", text="second")
        )
        stream_2.close()
        await task_2

        assert [m.content for m in conversation.messages] == ["first"]
        assert [m.content for m in new_conversation.messages] == ["second"]


class TestStartStop:
    async def test_start_then_stop_completes_cleanly(self, flow_factory) -> None:
        audio_output = AudioOutputStream()
        flow = flow_factory(audio_output=audio_output)

        await flow.start()
        await flow.stop()

        assert audio_output.closed()

    async def test_start_twice_raises(self, flow_factory) -> None:
        flow = flow_factory()

        await flow.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await flow.start()
        finally:
            await flow.stop()
