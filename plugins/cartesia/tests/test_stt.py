import asyncio

import pytest
import vision_agents.plugins.cartesia.stt as cartesia_stt_module

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.turn_detection import TurnEnded, TurnStarted
from vision_agents.plugins import cartesia


class TestCartesiaSTT:
    @pytest.fixture
    def participant(self) -> Participant:
        return Participant({}, user_id="test-user", id="test-user")

    async def test_start_duplicate_guard_runs_before_websocket_connect(
        self, monkeypatch
    ):
        class FakeConnection:
            def __init__(self) -> None:
                self.closed = False
                self._closed = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                await self._closed.wait()
                raise StopAsyncIteration

            async def close(self) -> None:
                self.closed = True
                self._closed.set()

        connections: list[FakeConnection] = []

        async def fake_connect(*args, **kwargs) -> FakeConnection:
            connection = FakeConnection()
            connections.append(connection)
            return connection

        monkeypatch.setattr(cartesia_stt_module.websockets, "connect", fake_connect)

        stt = cartesia.STT(api_key="fake")
        await stt.start()
        try:
            with pytest.raises(ValueError, match="already started"):
                await stt.start()
        finally:
            await stt.close()

        assert len(connections) == 1

    async def test_process_audio_fails_fast_after_listener_error(
        self, silence_1s_16khz
    ):
        class FailingConnection:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("socket broke")

        stt = cartesia.STT(api_key="fake")
        stt.started = True
        stt.connection = FailingConnection()
        stt._connection_ready.set()

        await stt._listen()

        assert stt.started is False
        assert stt.connection is None
        assert not stt._connection_ready.is_set()

        with pytest.raises(RuntimeError, match="websocket connection failed") as exc:
            await asyncio.wait_for(
                stt.process_audio(silence_1s_16khz),
                timeout=0.1,
            )

        assert isinstance(exc.value.__cause__, RuntimeError)
        assert str(exc.value.__cause__) == "socket broke"

    def test_defaults_to_latest_model_and_turn_detection(self):
        stt = cartesia.STT(api_key="fake")

        assert stt.model == "ink-2"
        assert stt.turn_detection is True
        assert stt.eager_turn_detection is True
        assert "model=ink-2" in stt._build_websocket_url()
        assert "cartesia_version=2026-03-01" in stt._build_websocket_url()

    async def test_handle_turn_events(self, participant):
        stt = cartesia.STT(api_key="fake")
        stt._current_participant = participant

        stt._handle_message({"type": "turn.start", "confidence": 0.8})
        stt._handle_message(
            {
                "type": "turn.update",
                "transcript": "hello",
                "words": [{"word": "hello", "confidence": 0.9}],
                "duration": 0.5,
            }
        )
        stt._handle_message(
            {
                "type": "turn.eager_end",
                "transcript": "hello world",
                "confidence": 0.7,
            }
        )
        stt._handle_message({"type": "turn.resume", "confidence": 0.6})
        stt._handle_message(
            {
                "type": "turn.end",
                "transcript": "hello world again",
                "confidence": 0.95,
                "duration_ms": 1200,
                "trailing_silence_ms": 250,
            }
        )

        items = await stt.output.collect(timeout=0)

        transcripts = [i for i in items if isinstance(i, Transcript)]
        assert [t.text for t in transcripts] == [
            "hello",
            "hello world",
            "hello world again",
        ]
        assert [t.final for t in transcripts] == [False, False, True]
        assert transcripts[0].confidence == 0.9
        assert transcripts[-1].model_name == "ink-2"

        turns_started = [i for i in items if isinstance(i, TurnStarted)]
        turns_ended = [i for i in items if isinstance(i, TurnEnded)]
        assert len(turns_started) == 2
        assert turns_started[0].confidence == 0.8
        assert turns_started[1].confidence == 0.6
        assert [t.eager for t in turns_ended] == [True, False]
        assert turns_ended[-1].duration_ms == 1200
        assert turns_ended[-1].trailing_silence_ms == 250

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self,
        mia_audio_48khz,
        silence_2s_48khz,
        participant,
        cartesia_api_key_required,
    ):
        stt = cartesia.STT(api_key=cartesia_api_key_required)
        await stt.start()
        try:
            await stt.process_audio(mia_audio_48khz, participant=participant)
            await stt.process_audio(silence_2s_48khz, participant=participant)

            items = await stt.output.collect(timeout=10.0)
        finally:
            await stt.close()

        transcripts = [i for i in items if isinstance(i, Transcript)]
        finals = [t for t in transcripts if t.final]
        assert finals, "No final Transcript emitted by Cartesia STT"
        full_transcript = " ".join(t.text for t in finals)
        assert "forgotten treasures" in full_transcript.lower()
        assert any(isinstance(i, TurnEnded) for i in items)
