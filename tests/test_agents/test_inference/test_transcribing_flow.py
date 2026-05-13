import asyncio

import numpy as np
import pytest
from getstream.video.rtc import PcmData
from getstream.video.rtc.track_util import AudioFormat
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.agents.events import (
    AgentTurnEndedEvent,
    AgentTurnStartedEvent,
    UserTranscriptEvent,
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
from vision_agents.core.agents.inference.transcribing_flow import (
    TranscribingInferenceFlow,
)
from vision_agents.core.agents.transcript import TranscriptStore
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.stt import STT
from vision_agents.core.stt.stt import Transcript, TranscriptResponse
from vision_agents.core.tts import TTS
from vision_agents.core.tts.tts import (
    TTSInput,
    TTSInputEnd,
    TTSOutputChunk,
    TTSOutputEnd,
)
from vision_agents.core.turn_detection import TurnDetector, TurnEnded, TurnStarted
from vision_agents.core.utils.stream import Stream

from .stubs import LLMStub, RealtimeStub, STTStub, TTSStub, TurnDetectorStub


def _dummy_pcm() -> PcmData:
    return PcmData(
        samples=np.zeros(0, dtype=np.int16),
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
        llm: LLM | None = None,
        tts: TTS | None = None,
        stt: STT | None = None,
        turn_detector: TurnDetector | None = None,
        audio_input: AudioInputStream | None = None,
        audio_output: AudioOutputStream | None = None,
    ) -> TranscribingInferenceFlow:
        return TranscribingInferenceFlow(
            audio_input=audio_input or AudioInputStream(),
            audio_output=audio_output or AudioOutputStream(),
            llm=llm or LLMStub(),
            stt=stt or STTStub(),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
            turn_detector=turn_detector,
            tts=tts,
        )

    return _build


class TestInit:
    async def test_init_flow_with_realtime_llm_fails(self, flow_factory):
        audio_output = AudioOutputStream()
        with pytest.raises(ValueError, match="Realtime"):
            flow_factory(audio_output=audio_output, llm=RealtimeStub())


class TestProcessSTTOutput:
    @pytest.fixture
    async def llm(self) -> LLMStub:
        return LLMStub.from_iterable(
            [
                LLMResponseDelta(delta="Hi ", item_id="m1", content_index=0),
                LLMResponseDelta(delta="there", item_id="m1", content_index=0),
                LLMResponseFinal(text="Hi there", item_id="m1"),
            ]
        )

    @pytest.fixture
    async def flow(
        self, llm, transcripts, conversation, events
    ) -> TranscribingInferenceFlow:
        return TranscribingInferenceFlow(
            audio_input=AudioInputStream(),
            audio_output=AudioOutputStream(),
            llm=llm,
            stt=STTStub(),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
        )

    @pytest.fixture
    async def flow_no_td(
        self, llm, transcripts, conversation, events
    ) -> TranscribingInferenceFlow:
        # Neither STT nor external TurnDetector drives turns — a final
        # Transcript is the commit signal on its own.
        return TranscribingInferenceFlow(
            audio_input=AudioInputStream(),
            audio_output=AudioOutputStream(),
            llm=llm,
            stt=STTStub(turn_detection=False),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
        )

    async def test_process_stt_output_final_transcript_turn_confirmed(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        user_started: list[UserTurnStartedEvent] = []
        user_ended: list[UserTurnEndedEvent] = []
        user_transcripts: list[UserTranscriptEvent] = []

        @flow.events.subscribe
        async def _on(
            event: UserTurnStartedEvent | UserTurnEndedEvent | UserTranscriptEvent,
        ):
            if isinstance(event, UserTurnStartedEvent):
                user_started.append(event)
            elif isinstance(event, UserTurnEndedEvent):
                user_ended.append(event)
            else:
                user_transcripts.append(event)

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        await flow.events.wait()
        items = llm_out.peek()

        deltas = [i for i in items if isinstance(i, LLMResponseDelta)]
        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(deltas) == 2
        assert len(finals) == 1
        assert finals[0].text == "Hi there"
        assert len(user_started) == 1
        assert user_started[0].participant is participant
        assert len(user_ended) == 1
        assert user_ended[0].participant is participant
        assert len(user_transcripts) == 1
        assert user_transcripts[0].text == "hello"
        assert user_transcripts[0].participant is participant

    async def test_process_stt_output_no_transcripts(self, flow, participant) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_whitespace_transcript(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="   ",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_partial_transcript_only(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_final_transcript_no_turn_end(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_eager_then_final_finalizes_once(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hello",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Eager end starts a speculative LLM turn for "hello".
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=True)
        )
        # Final transcript with same text — must NOT restart the turn.
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Non-eager end confirms the existing turn → finalize fires.
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        items = llm_out.peek()

        # One turn's worth of items: if finalize fired twice the count would double.
        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 1

    async def test_process_stt_output_eager_changed_transcript_finalizes_new(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Eager end with partial "hel" → speculative turn for "hel".
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=True)
        )
        # Final transcript differs → "hel" turn cancelled, new turn for "hello".
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        items = llm_out.peek()

        # Exactly one turn finalized — the cancelled "hel" turn must not leak.
        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 1

    async def test_process_stt_output_two_sequential_turns(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        # Turn 1
        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        # Collect the outputs of the first turn before starting a new one (it may cancel the current one)
        items = await llm_out.collect(timeout=2.0)
        # Turn 2
        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="world",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage

        # Collect the outputs of the second turn
        items += await llm_out.collect(timeout=2.0)

        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 2

    async def test_process_stt_output_barge_in_cancels_in_flight_turn(
        self, flow, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        # Final transcript starts a non-eager turn; not yet confirmed.
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Barge-in before TurnEnded — cancels the in-flight turn.
        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        # Follow-up turn to prove state reset and the pipeline still works.
        await stt_out.send(
            Transcript(
                text="world",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        stt_out.close()
        await stage
        items = llm_out.peek()

        # Only the follow-up turn reached finalize. If the "hello" turn had
        # leaked through we'd see two finals.
        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 1

    async def test_process_stt_output_turn_started_flushes_buffered_transcripts(
        self, flow, transcripts, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="stale",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Barge-in — interrupt() must flush the pending buffer for this user.
        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))

        stt_out.close()
        await stage

        buffer = transcripts.get_buffer(
            participant_id=participant.id, user_id=participant.user_id
        )
        assert buffer is None

    async def test_process_stt_output_replacement_mode_overwrites_previous_partial(
        self, flow, transcripts, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            Transcript(
                text="hello",
                mode="replacement",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage

        buffer = transcripts.get_buffer(
            participant_id=participant.id, user_id=participant.user_id
        )
        assert buffer is not None
        # Replacement mode: second partial overwrites the first, not concat.
        assert buffer.text == "hello"

    async def test_process_stt_output_no_td_final_transcript_auto_commits(
        self, flow_no_td, participant
    ) -> None:
        # Without a turn-detection source, a final Transcript commits on its own.
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow_no_td.process_stt_output(stt_out, llm_out))

        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        items = llm_out.peek()

        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 1
        assert finals[0].text == "Hi there"

    async def test_process_stt_output_no_td_partial_only_does_not_commit(
        self, flow_no_td, participant
    ) -> None:
        # A partial without a trailing final must not commit — auto-confirm
        # is gated on buffer.final.
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow_no_td.process_stt_output(stt_out, llm_out))

        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_no_td_whitespace_final_does_not_auto_commit(
        self, flow_no_td, participant
    ) -> None:
        # In no-turn-detection mode a final Transcript is the commit signal,
        # so without the empty-transcript guard a whitespace-only final would
        # auto-start an LLM turn with transcript="".
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow_no_td.process_stt_output(stt_out, llm_out))

        await stt_out.send(
            Transcript(
                text="   ",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        assert not llm_out.peek()

    async def test_process_stt_output_no_td_partial_then_final_fires_once(
        self, flow_no_td, participant
    ) -> None:
        # A preceding partial must not cause a spurious extra commit — only
        # the final Transcript triggers the LLM turn.
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow_no_td.process_stt_output(stt_out, llm_out))

        await stt_out.send(
            Transcript(
                text="hel",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        items = llm_out.peek()

        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 1

    async def test_process_stt_output_no_td_two_sequential_finals(
        self, flow_no_td, participant
    ) -> None:
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        stage = asyncio.create_task(flow_no_td.process_stt_output(stt_out, llm_out))

        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        items = await llm_out.collect(2.0)

        await stt_out.send(
            Transcript(
                text="world",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )

        stt_out.close()
        await stage
        items += await llm_out.collect(2.0)

        finals = [i for i in items if isinstance(i, LLMResponseFinal)]
        assert len(finals) == 2

    async def test_simple_response_then_process_stt_output_no_td(
        self, transcripts, conversation, participant, events
    ) -> None:
        """After ``simple_response`` returns, ``process_stt_output`` still runs a no-TD final."""

        async def llm(text: str, _participant: Participant | None = None):
            if text == "inject":
                yield LLMResponseFinal(text="inj", item_id="i1")
            else:
                yield LLMResponseFinal(text="stt", item_id="s1")

        flow = TranscribingInferenceFlow(
            audio_input=AudioInputStream(),
            audio_output=AudioOutputStream(),
            llm=LLMStub.from_callable(llm),
            stt=STTStub(turn_detection=False),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
        )
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        await flow.simple_response("inject", participant)
        await asyncio.sleep(0)

        task = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        stt_out.close()
        await task

        finals = [i for i in llm_out.peek() if isinstance(i, LLMResponseFinal)]
        assert [f.text for f in finals] == ["stt"]

    async def test_simple_response_then_turn_ended_before_final_with_turn_detector(
        self, transcripts, conversation, participant, events
    ) -> None:
        """TurnDetector path: ``TurnEnded`` arrives before the final STT transcript."""

        async def llm(text: str, _participant: Participant | None = None):
            if text == "inject":
                yield LLMResponseFinal(text="inj", item_id="i1")
            else:
                yield LLMResponseFinal(text="td", item_id="m2")

        flow = TranscribingInferenceFlow(
            audio_input=AudioInputStream(),
            audio_output=AudioOutputStream(),
            llm=LLMStub.from_callable(llm),
            stt=STTStub(turn_detection=False),
            turn_detector=TurnDetectorStub(),
            transcripts=transcripts,
            agent_user_id="agent-1",
            conversation=conversation,
            events=events,
        )
        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        await flow.simple_response("inject", participant)
        await asyncio.sleep(0)

        task = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )
        await stt_out.send(
            Transcript(
                text="hello",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        stt_out.close()
        await task

        finals = [i for i in llm_out.peek() if isinstance(i, LLMResponseFinal)]
        assert [f.text for f in finals] == ["td"]


class TestInterrupt:
    async def test_interrupt_signals_downstream_via_audio_output_flush(
        self, flow_factory
    ) -> None:
        audio_output = AudioOutputStream()
        flow = flow_factory(audio_output=audio_output)

        await flow.interrupt()

        assert audio_output.peek() == [AudioOutputFlush()]

    async def test_interrupt_flushes_buffered_user_transcripts(
        self, flow_factory, transcripts, participant
    ) -> None:
        flow = flow_factory()
        transcripts.update_user_transcript(
            participant_id=participant.id,
            user_id=participant.user_id,
            text="hel",
            mode="replacement",
        )
        assert (
            transcripts.get_buffer(
                participant_id=participant.id, user_id=participant.user_id
            )
            is not None
        )

        await flow.interrupt()

        assert (
            transcripts.get_buffer(
                participant_id=participant.id, user_id=participant.user_id
            )
            is None
        )

    async def test_interrupt_bumps_tts_epoch_when_tts_set(self, flow_factory) -> None:
        tts = TTSStub()
        flow = flow_factory(tts=tts)
        epoch_before = tts.epoch

        await flow.interrupt()

        assert tts.epoch == epoch_before + 1

    async def test_interrupt_without_tts_does_not_raise(self, flow_factory) -> None:
        flow = flow_factory(tts=None)

        await flow.interrupt()

    async def test_interrupt_drops_pre_queued_audio_output_and_appends_flush(
        self, flow_factory
    ) -> None:
        audio_output = AudioOutputStream()
        flow = flow_factory(audio_output=audio_output)
        audio_output.send_nowait(AudioOutputChunk(data=None, final=False))

        await flow.interrupt()

        # Pre-queued chunk is dropped (clear() ran) and a single
        # AudioOutputFlush is appended after — proves clear/flush ordering.
        assert audio_output.peek() == [AudioOutputFlush()]

    async def test_interrupt_is_idempotent(self, flow_factory) -> None:
        tts = TTSStub()
        audio_output = AudioOutputStream()
        flow = flow_factory(tts=tts, audio_output=audio_output)

        await flow.interrupt()
        await flow.interrupt()

        # Each call clears the queue before appending its own flush marker,
        # so the final state always holds exactly one marker.
        assert audio_output.peek() == [AudioOutputFlush()]
        assert tts.epoch == 2

    async def test_interrupt_drops_in_flight_llm_output(
        self, flow_factory, participant
    ) -> None:
        blocker = asyncio.Event()

        async def blocking(_text, _participant):
            await blocker.wait()
            yield LLMResponseFinal(text="leaked", item_id="m1")

        llm = LLMStub.from_callable(blocking)
        flow = flow_factory(llm=llm)

        stt_out: Stream[TurnStarted | Transcript | TurnEnded] = Stream()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        stage = asyncio.create_task(flow.process_stt_output(stt_out, llm_out))

        # Start an in-flight, unconfirmed LLM turn that hangs on the blocker.
        await stt_out.send(TurnStarted(participant=participant, confidence=0.5))
        await stt_out.send(
            Transcript(
                text="hi",
                mode="final",
                participant=participant,
                response=TranscriptResponse(),
            )
        )
        # Yield so the LLM turn's task suspends on blocker.wait().
        await asyncio.sleep(0)

        await flow.interrupt()

        # Release the (now-cancelled) LLM task — its yield must not leak.
        blocker.set()

        stt_out.close()
        await stage

        assert llm_out.peek() == []


class TestProcessTurnDetection:
    async def test_forwards_events_to_stt_output(
        self, flow_factory, participant
    ) -> None:
        flow = flow_factory()
        td_out: Stream[TurnStarted | TurnEnded] = Stream()
        stt_out: Stream[TurnStarted | TurnEnded | Transcript] = Stream()

        stage = asyncio.create_task(flow.process_turn_detection(td_out, stt_out))

        started = TurnStarted(participant=participant, confidence=0.5)
        ended = TurnEnded(participant=participant, confidence=0.9, eager=False)
        await td_out.send(started)
        await td_out.send(ended)

        td_out.close()
        await stage

        assert stt_out.peek() == [started, ended]

    async def test_swallows_downstream_send_failure_and_keeps_iterating(
        self, flow_factory, participant
    ) -> None:
        flow = flow_factory()
        td_out: Stream[TurnStarted | TurnEnded] = Stream()
        stt_out: Stream[TurnStarted | TurnEnded | Transcript] = Stream()
        # Closing stt_out makes the forwarding send() raise StreamClosed.
        stt_out.close()

        stage = asyncio.create_task(flow.process_turn_detection(td_out, stt_out))

        await td_out.send(TurnStarted(participant=participant, confidence=0.5))
        await td_out.send(
            TurnEnded(participant=participant, confidence=0.9, eager=False)
        )

        td_out.close()
        # If log_exceptions did not swallow StreamClosed, awaiting the task
        # would propagate it here.
        await stage


class TestProcessInputAudio:
    async def test_routes_chunks_to_stt_without_turn_detector(
        self, flow_factory, participant
    ) -> None:
        stt = STTStub(
            responses=[
                Transcript(
                    text="a",
                    mode="delta",
                    participant=participant,
                    response=TranscriptResponse(),
                ),
                Transcript(
                    text="b",
                    mode="delta",
                    participant=participant,
                    response=TranscriptResponse(),
                ),
            ],
        )
        audio_input = AudioInputStream()
        flow = flow_factory(stt=stt, audio_input=audio_input)

        stage = asyncio.create_task(flow.process_audio_input(audio_input))

        await audio_input.send(
            AudioInputChunk(data=_dummy_pcm(), participant=participant)
        )
        await audio_input.send(
            AudioInputChunk(data=_dummy_pcm(), participant=participant)
        )

        audio_input.close()
        await stage

        items = stt.output.peek()
        assert [i.text for i in items] == ["a", "b"]

    async def test_routes_chunks_to_both_turn_detector_and_stt(
        self, flow_factory, participant
    ) -> None:
        # turn_detection=False so the external TurnDetector is not overridden.
        stt = STTStub(
            turn_detection=False,
            responses=[
                Transcript(
                    text=f"t{i}",
                    mode="delta",
                    participant=participant,
                    response=TranscriptResponse(),
                )
                for i in range(2)
            ],
        )
        td = TurnDetectorStub(
            responses=[TurnStarted(participant=participant, confidence=0.5)] * 2
        )
        audio_input = AudioInputStream()
        flow = flow_factory(stt=stt, turn_detector=td, audio_input=audio_input)

        stage = asyncio.create_task(flow.process_audio_input(audio_input))

        for _ in range(2):
            await audio_input.send(
                AudioInputChunk(data=_dummy_pcm(), participant=participant)
            )

        audio_input.close()
        await stage

        stt_items = stt.output.peek()
        td_items = td.output.peek()
        assert [i.text for i in stt_items] == ["t0", "t1"]
        assert len(td_items) == 2

    async def test_continues_after_stt_failure_on_one_chunk(
        self, flow_factory, participant
    ) -> None:
        first_call = True

        async def failing_once(pcm_data, participant):
            nonlocal first_call
            if first_call:
                first_call = False
                raise RuntimeError("boom")
            yield Transcript(
                text="ok",
                mode="delta",
                participant=participant,
                response=TranscriptResponse(),
            )

        stt = STTStub.from_callable(failing_once)
        audio_input = AudioInputStream()
        flow = flow_factory(stt=stt, audio_input=audio_input)

        stage = asyncio.create_task(flow.process_audio_input(audio_input))

        # First chunk causes STT to raise; log_exceptions swallows it.
        await audio_input.send(
            AudioInputChunk(data=_dummy_pcm(), participant=participant)
        )
        # Second chunk must still be processed — proves the loop kept going.
        await audio_input.send(
            AudioInputChunk(data=_dummy_pcm(), participant=participant)
        )

        audio_input.close()
        await stage

        items = stt.output.peek()
        assert [i.text for i in items] == ["ok"]


class TestProcessLLMOutput:
    async def test_delta_emits_tts_delta_and_upserts_partial_assistant_message(
        self, flow_factory, conversation
    ) -> None:
        flow = flow_factory()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()

        stage = asyncio.create_task(flow.process_llm_output(llm_out, tts_in))

        await llm_out.send(LLMResponseDelta(delta="Hi ", item_id="m1", content_index=0))

        llm_out.close()
        await stage

        assert tts_in.peek() == [TTSInput(text="Hi ", delta=True)]
        assistant = [m for m in conversation.messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].content == "Hi "

    async def test_final_emits_tts_full_text_then_end_and_completes_message(
        self, flow_factory, conversation
    ) -> None:
        flow = flow_factory()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()

        stage = asyncio.create_task(flow.process_llm_output(llm_out, tts_in))

        await llm_out.send(LLMResponseFinal(text="Hi there", item_id="m1"))

        llm_out.close()
        await stage

        assert tts_in.peek() == [
            TTSInput(text="Hi there", delta=False),
            TTSInputEnd(),
        ]
        assistant = [m for m in conversation.messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].content == "Hi there"

    async def test_delta_sanitized_in_both_conversation_and_tts(
        self, flow_factory, conversation
    ) -> None:
        flow = flow_factory()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()

        stage = asyncio.create_task(flow.process_llm_output(llm_out, tts_in))

        await llm_out.send(
            LLMResponseDelta(delta="*Hi*", item_id="m1", content_index=0)
        )

        llm_out.close()
        await stage

        assert tts_in.peek() == [TTSInput(text="Hi", delta=True)]
        assistant = [m for m in conversation.messages if m.role == "assistant"]
        assert assistant[0].content == "Hi"

    async def test_final_sanitizes_conversation_but_not_tts_input(
        self, flow_factory, conversation
    ) -> None:
        # For LLMResponseFinal the TTS receives the unsanitized item.text;
        # only the conversation gets the sanitized version.
        flow = flow_factory()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()

        stage = asyncio.create_task(flow.process_llm_output(llm_out, tts_in))

        await llm_out.send(LLMResponseFinal(text="#Hello", item_id="m1"))

        llm_out.close()
        await stage

        assert tts_in.peek() == [
            TTSInput(text="#Hello", delta=False),
            TTSInputEnd(),
        ]
        assistant = [m for m in conversation.messages if m.role == "assistant"]
        assert assistant[0].content == "Hello"

    async def test_delta_sequence_then_final_produces_expected_stream(
        self, flow_factory, conversation
    ) -> None:
        flow = flow_factory()
        llm_out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()

        stage = asyncio.create_task(flow.process_llm_output(llm_out, tts_in))

        await llm_out.send(
            LLMResponseDelta(delta="Hello ", item_id="m1", content_index=0)
        )
        await llm_out.send(
            LLMResponseDelta(delta="world", item_id="m1", content_index=1)
        )
        await llm_out.send(LLMResponseFinal(text="Hello world!", item_id="m1"))

        llm_out.close()
        await stage

        assert tts_in.peek() == [
            TTSInput(text="Hello ", delta=True),
            TTSInput(text="world", delta=True),
            TTSInput(text="Hello world!", delta=False),
            TTSInputEnd(),
        ]
        assistant = [m for m in conversation.messages if m.role == "assistant"]
        assert len(assistant) == 1
        # Final replaces the accumulated delta content.
        assert assistant[0].content == "Hello world!"


class TestProcessTTS:
    async def test_streaming_partial_sentence_does_not_emit(self, flow_factory) -> None:
        tts = TTSStub(streaming=True)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="Hi", delta=True))

        tts_in.close()
        await stage

        assert tts_out.peek() == []

    async def test_streaming_completed_sentence_emits_sentence_chunk(
        self, flow_factory
    ) -> None:
        tts = TTSStub(streaming=True)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="Hello. ", delta=True))

        tts_in.close()
        await stage

        chunks = tts_out.peek()
        assert [c.text for c in chunks] == ["Hello."]

    async def test_streaming_input_end_flushes_remainder(self, flow_factory) -> None:
        tts = TTSStub(streaming=True)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="Hi", delta=True))
        await tts_in.send(TTSInputEnd())

        tts_in.close()
        await stage

        chunks = tts_out.peek()
        assert [c.text for c in chunks] == ["Hi"]

    async def test_streaming_ignores_delta_false_inputs(self, flow_factory) -> None:
        tts = TTSStub(streaming=True)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="ignored", delta=False))

        tts_in.close()
        await stage

        assert tts_out.peek() == []

    async def test_non_streaming_emits_only_on_full_utterance(
        self, flow_factory
    ) -> None:
        tts = TTSStub(streaming=False)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="delta", delta=True))
        await tts_in.send(TTSInput(text="full", delta=False))
        await tts_in.send(TTSInputEnd())

        tts_in.close()
        await stage

        chunks = tts_out.peek()
        assert [c.text for c in chunks] == ["full"]

    async def test_non_streaming_passes_through_all_chunks(self, flow_factory) -> None:
        preloaded = [
            TTSOutputChunk(text="c0", index=0),
            TTSOutputChunk(text="c1", index=1),
            TTSOutputChunk(text="c2", index=2, final=True),
        ]
        tts = TTSStub(chunks=preloaded, streaming=False)
        flow = flow_factory(tts=tts)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        stage = asyncio.create_task(flow.process_tts(tts_in, tts_out))

        await tts_in.send(TTSInput(text="x", delta=False))

        tts_in.close()
        await stage

        assert tts_out.peek() == preloaded

    async def test_returns_early_when_tts_is_none(self, flow_factory) -> None:
        flow = flow_factory(tts=None)
        tts_in: Stream[TTSInput | TTSInputEnd] = Stream()
        tts_out: Stream[TTSOutputChunk] = Stream()

        # Pre-queue an item; process_tts must not consume it when tts is None.
        tts_in.send_nowait(TTSInput(text="x", delta=False))

        await flow.process_tts(tts_in, tts_out)

        # Item still present — the method returned without reading.
        assert tts_in.get_nowait() == TTSInput(text="x", delta=False)
        assert tts_out.peek() == []


class TestWriteAudioOutput:
    async def test_forwards_chunks_preserving_final_flag(self, flow_factory) -> None:
        flow = flow_factory()
        started: list[AgentTurnStartedEvent] = []
        ended: list[AgentTurnEndedEvent] = []

        @flow.events.subscribe
        async def _on(event: AgentTurnStartedEvent | AgentTurnEndedEvent):
            (started if isinstance(event, AgentTurnStartedEvent) else ended).append(
                event
            )

        tts_out: Stream[TTSOutputChunk | TTSOutputEnd] = Stream()
        audio_out = AudioOutputStream()

        stage = asyncio.create_task(flow.write_audio_output(tts_out, audio_out))

        # data=None lets the chunk pass through AudioOutputStream unchanged,
        # bypassing its 20ms re-chunking so the assertion stays focused.
        await tts_out.send(TTSOutputChunk(data=None, final=False))
        await tts_out.send(TTSOutputChunk(data=None, final=True))

        tts_out.close()
        await stage
        await flow.events.wait()

        assert audio_out.peek() == [
            AudioOutputChunk(data=None, final=False),
            AudioOutputChunk(data=None, final=True),
        ]
        assert len(started) == 1
        assert len(ended) == 1
        assert ended[0].interrupted is False

    async def test_tts_output_end_emits_interrupted_agent_turn_ended(
        self, flow_factory
    ) -> None:
        flow = flow_factory()
        seen: list[AgentTurnEndedEvent] = []

        @flow.events.subscribe
        async def _on(event: AgentTurnEndedEvent):
            seen.append(event)

        tts_out: Stream[TTSOutputChunk | TTSOutputEnd] = Stream()
        audio_out = AudioOutputStream()
        stage = asyncio.create_task(flow.write_audio_output(tts_out, audio_out))

        await tts_out.send(TTSOutputChunk(data=None, final=False))
        await tts_out.send(TTSOutputEnd(interrupted=True))
        tts_out.close()
        await stage
        await flow.events.wait()

        assert len(seen) == 1
        assert seen[0].interrupted is True

    async def test_tts_output_end_without_started_emits_nothing(
        self, flow_factory
    ) -> None:
        flow = flow_factory()
        seen: list[AgentTurnEndedEvent] = []

        @flow.events.subscribe
        async def _on(event: AgentTurnEndedEvent):
            seen.append(event)

        tts_out: Stream[TTSOutputChunk | TTSOutputEnd] = Stream()
        audio_out = AudioOutputStream()
        stage = asyncio.create_task(flow.write_audio_output(tts_out, audio_out))

        await tts_out.send(TTSOutputEnd(interrupted=True))
        tts_out.close()
        await stage
        await flow.events.wait()

        assert seen == []
