import pytest

from vision_agents.core.edge.types import Participant
from vision_agents.core.stt.events import STTTranscriptEvent, STTPartialTranscriptEvent
from vision_agents.plugins import assemblyai
from conftest import STTSession


class TestAssemblyAISTT:
    """Integration tests for AssemblyAI STT."""

    @pytest.fixture
    async def stt(self):
        """Create and manage AssemblyAI STT lifecycle."""
        stt = assemblyai.STT()
        try:
            await stt.start()
            yield stt
        finally:
            await stt.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(
        self, stt, mia_audio_48khz, silence_2s_48khz
    ):
        session = STTSession(stt)

        await stt.process_audio(
            mia_audio_48khz, participant=Participant({}, user_id="hi", id="hi")
        )

        await stt.process_audio(
            silence_2s_48khz, participant=Participant({}, user_id="hi", id="hi")
        )

        await session.wait_for_result(timeout=30.0)
        assert not session.errors

        full_transcript = session.get_full_transcript()
        assert full_transcript is not None
        assert "forgotten treasures" in full_transcript.lower()

        assert session.transcripts[0].participant.user_id == "hi"

    async def test_prompt_and_keyterms_exclusive(self):
        with pytest.raises(ValueError, match="cannot be used together"):
            assemblyai.STT(
                api_key="test",
                prompt="test prompt",
                keyterms_prompt=["term1"],
            )

    async def test_default_configuration(self):
        stt = assemblyai.STT(api_key="test-key")
        assert stt._speech_model == "u3-rt-pro"
        assert stt._sample_rate == 16000
        assert stt.turn_detection is True
        assert stt.provider_name == "assemblyai"

    async def test_reconnect_defaults(self):
        stt = assemblyai.STT(api_key="test-key")
        assert stt._max_reconnect_attempts == 3
        assert stt._reconnect_backoff_initial_s == 0.5
        assert stt._reconnect_backoff_max_s == 4.0

    async def test_custom_reconnect_config(self):
        stt = assemblyai.STT(
            api_key="test-key",
            max_reconnect_attempts=5,
            reconnect_backoff_initial_s=1.0,
            reconnect_backoff_max_s=8.0,
        )
        assert stt._max_reconnect_attempts == 5
        assert stt._reconnect_backoff_initial_s == 1.0
        assert stt._reconnect_backoff_max_s == 8.0

    async def test_speaker_labels_default_disabled(self):
        stt = assemblyai.STT(api_key="test-key")
        assert stt._speaker_labels_enabled is False
        assert stt._max_speakers is None
        assert stt._speaker_participants == {}

    async def test_speaker_labels_enabled(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)
        assert stt._speaker_labels_enabled is True

    async def test_speaker_labels_with_max_speakers(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True, max_speakers=3)
        assert stt._speaker_labels_enabled is True
        assert stt._max_speakers == 3

    async def test_max_speakers_without_speaker_labels_raises(self):
        with pytest.raises(ValueError, match="max_speakers requires speaker_labels=True"):
            assemblyai.STT(api_key="test-key", max_speakers=2)

    async def test_build_ws_url_without_speaker_labels(self):
        stt = assemblyai.STT(api_key="test-key")
        url = stt._build_ws_url()
        assert "speaker_labels" not in url
        assert "max_speakers" not in url

    async def test_build_ws_url_with_speaker_labels(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)
        url = stt._build_ws_url()
        assert "speaker_labels=true" in url
        assert "max_speakers" not in url

    async def test_build_ws_url_with_speaker_labels_and_max_speakers(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True, max_speakers=4)
        url = stt._build_ws_url()
        assert "speaker_labels=true" in url
        assert "max_speakers=4" in url

    async def test_resolve_participant_without_diarization(self):
        stt = assemblyai.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        result = stt._resolve_participant("A")
        assert result is participant

    async def test_resolve_participant_with_null_label(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        result = stt._resolve_participant(None)
        assert result is participant

    async def test_resolve_participant_creates_synthetic(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        result = stt._resolve_participant("A")
        assert result is not None
        assert result.user_id == "speaker_A"
        assert result.id.startswith("speaker_A_")
        assert result.original is None

    async def test_resolve_participant_caches_synthetic(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        first = stt._resolve_participant("B")
        second = stt._resolve_participant("B")
        assert first is second

    async def test_resolve_participant_distinct_per_label(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        a = stt._resolve_participant("A")
        b = stt._resolve_participant("B")
        assert a is not b
        assert a.user_id == "speaker_A"
        assert b.user_id == "speaker_B"
        assert a.id != b.id

    async def test_handle_turn_with_speaker_label_emits_correct_participant(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        events: list[STTTranscriptEvent] = []

        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            events.append(event)

        stt._handle_turn({
            "transcript": "Hello world",
            "end_of_turn": True,
            "speaker_label": "A",
        })
        await stt.events.wait()

        assert len(events) == 1
        assert events[0].text == "Hello world"
        assert events[0].participant.user_id == "speaker_A"
        assert events[0].response.other == {"speaker_label": "A"}

    async def test_handle_turn_partial_with_speaker_label(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        events: list[STTPartialTranscriptEvent] = []

        @stt.events.subscribe
        async def on_partial(event: STTPartialTranscriptEvent):
            events.append(event)

        stt._handle_turn({
            "transcript": "Hello",
            "end_of_turn": False,
            "speaker_label": "B",
        })
        await stt.events.wait()

        assert len(events) == 1
        assert events[0].text == "Hello"
        assert events[0].participant.user_id == "speaker_B"
        assert events[0].response.other == {"speaker_label": "B"}

    async def test_handle_turn_without_diarization_no_other_field(self):
        stt = assemblyai.STT(api_key="test-key")
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        events: list[STTTranscriptEvent] = []

        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            events.append(event)

        stt._handle_turn({
            "transcript": "Hello world",
            "end_of_turn": True,
        })
        await stt.events.wait()

        assert len(events) == 1
        assert events[0].response.other is None
        assert events[0].participant is participant

    async def test_handle_turn_null_speaker_label_falls_back(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)
        participant = Participant(original=None, user_id="user1", id="p1")
        stt._current_participant = participant

        events: list[STTTranscriptEvent] = []

        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            events.append(event)

        stt._handle_turn({
            "transcript": "Short",
            "end_of_turn": True,
            "speaker_label": None,
        })
        await stt.events.wait()

        assert len(events) == 1
        assert events[0].participant is participant
        assert events[0].response.other == {"speaker_label": None}

    async def test_handle_turn_multi_speaker_conversation(self):
        stt = assemblyai.STT(api_key="test-key", speaker_labels=True)

        events: list[STTTranscriptEvent] = []

        @stt.events.subscribe
        async def on_transcript(event: STTTranscriptEvent):
            events.append(event)

        stt._handle_turn({
            "transcript": "Good morning",
            "end_of_turn": True,
            "speaker_label": "A",
        })
        stt._handle_turn({
            "transcript": "Good morning, how are you?",
            "end_of_turn": True,
            "speaker_label": "B",
        })
        stt._handle_turn({
            "transcript": "I'm doing well, thanks",
            "end_of_turn": True,
            "speaker_label": "A",
        })
        await stt.events.wait()

        assert len(events) == 3
        assert events[0].participant.user_id == "speaker_A"
        assert events[1].participant.user_id == "speaker_B"
        assert events[2].participant.user_id == "speaker_A"
        assert events[0].participant is events[2].participant
