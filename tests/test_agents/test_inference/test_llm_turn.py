import asyncio

import pytest
from vision_agents.core.agents.inference.llm_turn import LLMTurn
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.stream import Stream

from .stubs import LLMStub


@pytest.fixture
def participant() -> Participant:
    return Participant(id="p1", user_id="u1", original=None)


class TestLLMTurn:
    async def test_happy_path_forwards_sequence_to_output(
        self, participant: Participant
    ) -> None:
        items = [
            LLMResponseDelta(delta="Hi ", item_id="m1", content_index=0),
            LLMResponseDelta(delta="there", item_id="m1", content_index=0),
            LLMResponseFinal(text="Hi there", item_id="m1"),
        ]
        llm = LLMStub.from_iterable(items)
        turn = LLMTurn(transcript="user said this", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        turn.start(llm)
        assert turn.started
        turn.confirm()
        assert turn.confirmed
        turn.finalize(out)

        assert await out.collect(timeout=2.0) == items

        assert not turn.cancelled

    async def test_start_twice_raises(self, participant: Participant) -> None:
        llm = LLMStub.from_iterable(
            [LLMResponseFinal(text="x", item_id="m1")],
        )
        turn = LLMTurn(transcript="hi", participant=participant)
        turn.start(llm)
        with pytest.raises(RuntimeError, match="already running"):
            turn.start(llm)

    async def test_finalize_before_start_raises(self, participant: Participant) -> None:
        turn = LLMTurn(transcript="hi", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        with pytest.raises(RuntimeError, match="started first"):
            turn.finalize(out)

    async def test_finalize_before_confirm_raises(
        self, participant: Participant
    ) -> None:
        llm = LLMStub.from_iterable(
            [LLMResponseFinal(text="x", item_id="m1")],
        )
        turn = LLMTurn(transcript="hi", participant=participant)
        turn.start(llm)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        with pytest.raises(RuntimeError, match="confirmed first"):
            turn.finalize(out)

    def test_confirm_twice_raises(self, participant: Participant) -> None:
        turn = LLMTurn(transcript="hi", participant=participant)
        turn.confirm()
        with pytest.raises(RuntimeError, match="already confirmed"):
            turn.confirm()

    async def test_simple_response_receives_transcript_and_participant(
        self, participant: Participant
    ) -> None:
        seen: list[tuple[str, Participant | None]] = []

        async def capture(text: str, p: Participant | None):
            seen.append((text, p))
            yield LLMResponseFinal(text="ok", item_id="m1")

        expected = [LLMResponseFinal(text="ok", item_id="m1")]
        llm = LLMStub.from_callable(capture)
        turn = LLMTurn(transcript="exact", participant=participant)

        turn.start(llm)
        turn.confirm()
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        turn.finalize(out)

        assert await out.collect(timeout=2.0) == expected

        assert seen == [("exact", participant)]

    async def test_cancel_mid_llm_clears_started_and_skips_output(
        self, participant: Participant
    ) -> None:
        blocker = asyncio.Event()

        async def hanging(_text: str, _participant: Participant | None):
            await blocker.wait()
            yield LLMResponseFinal(text="leaked", item_id="m1")

        llm = LLMStub.from_callable(hanging)
        turn = LLMTurn(transcript="hi", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        turn.start(llm)
        await asyncio.sleep(0)

        await turn.cancel()
        blocker.set()
        await asyncio.sleep(0)

        assert turn.cancelled
        assert not turn.started
        assert await out.collect(timeout=0) == []

    async def test_cancel_mid_llm_signals_interrupt_to_llm(
        self, participant: Participant
    ) -> None:
        # Mirrors how real plugins treat interrupt() as a stop signal
        # (e.g. transformers_llm flips a stopping_criteria flag).
        class InterruptingLLM(LLMStub):
            interrupted: bool = False

            async def interrupt(self) -> None:
                self.interrupted = True

        blocker = asyncio.Event()

        async def hanging(_text: str, _participant: Participant | None):
            await blocker.wait()
            yield LLMResponseFinal(text="leaked", item_id="m1")

        llm = InterruptingLLM(factory=hanging)
        turn = LLMTurn(transcript="hi", participant=participant)

        turn.start(llm)
        await asyncio.sleep(0)
        assert not llm.interrupted

        await turn.cancel()

        assert llm.interrupted

    async def test_cancel_before_start_is_no_op(self, participant: Participant) -> None:
        turn = LLMTurn(transcript="hi", participant=participant)
        await turn.cancel()
        assert not turn.cancelled
        assert not turn.started

    async def test_cancel_after_completed_pipeline_does_not_raise(
        self, participant: Participant
    ) -> None:
        expected = [LLMResponseFinal(text="x", item_id="m1")]
        llm = LLMStub.from_iterable(expected)
        turn = LLMTurn(transcript="hi", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()
        turn.start(llm)
        turn.confirm()
        turn.finalize(out)

        assert await out.collect(timeout=2.0) == expected

        await turn.cancel()

    async def test_finalize_twice_does_not_duplicate_output(
        self, participant: Participant
    ) -> None:
        items = [
            LLMResponseDelta(delta="a", item_id="m1", content_index=0),
            LLMResponseFinal(text="a", item_id="m1"),
        ]
        llm = LLMStub.from_iterable(items)
        turn = LLMTurn(transcript="hi", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        turn.start(llm)
        turn.confirm()
        turn.finalize(out)
        turn.finalize(out)

        assert await out.collect(timeout=2.0) == items

    async def test_final_response_records_metric(
        self, participant: Participant
    ) -> None:
        items = [
            LLMResponseDelta(delta="Hi", item_id="m1", content_index=0),
            LLMResponseFinal(
                text="Hi",
                item_id="m1",
                latency_ms=42.0,
                time_to_first_token_ms=7.0,
                input_tokens=10,
                output_tokens=20,
                model="test-model",
            ),
        ]
        llm = LLMStub.from_iterable(items)
        turn = LLMTurn(transcript="hi", participant=participant)
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        turn.start(llm)
        turn.confirm()
        turn.finalize(out)
        await out.collect(timeout=2.0)

        m = llm.metrics.agent_metrics
        assert m.llm_input_tokens__total.value() == 10
        assert m.llm_output_tokens__total.value() == 20
        assert m.llm_latency_ms__avg.value() == 42.0
        assert m.llm_time_to_first_token_ms__avg.value() == 7.0

    async def test_output_stays_empty_until_finalize_then_receives_response(
        self, participant: Participant
    ) -> None:
        llm_done = asyncio.Event()
        expected = [LLMResponseFinal(text="done", item_id="m1")]
        out: Stream[LLMResponseDelta | LLMResponseFinal] = Stream()

        async def emit(_text: str, _participant: Participant | None):
            yield LLMResponseFinal(text="done", item_id="m1")
            llm_done.set()

        llm = LLMStub.from_callable(emit)
        turn = LLMTurn(transcript="hi", participant=participant)
        turn.start(llm)
        turn.confirm()

        await asyncio.wait_for(llm_done.wait(), timeout=2.0)
        assert await out.collect(timeout=0) == []

        turn.finalize(out)
        assert await out.collect(timeout=2.0) == expected
