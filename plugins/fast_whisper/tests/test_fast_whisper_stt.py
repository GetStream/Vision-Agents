import asyncio

import pytest
from dotenv import load_dotenv

from vision_agents.plugins import fast_whisper
from conftest import STTSession

# Load environment variables
load_dotenv()


class TestFastWhisperSTT:
    """Integration tests for Fast Whisper STT"""

    @pytest.fixture
    async def stt(self):
        """Create and manage Fast Whisper STT lifecycle"""
        stt_instance = fast_whisper.STT(model_size="tiny")  # Use tiny for faster tests
        try:
            yield stt_instance
        finally:
            await stt_instance.close()

    @pytest.mark.integration
    async def test_transcribe_mia_audio(self, stt, mia_audio_16khz):
        """Test transcription of 16kHz audio."""
        # Warm up the model
        await stt.warmup()
        
        # Create session to collect transcripts and errors
        session = STTSession(stt)
        
        # Process the audio
        await stt.process_audio(mia_audio_16khz)
        
        # Wait for result
        await session.wait_for_result(timeout=60.0)
        assert not session.errors
        
        # Verify transcript
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
        assert "forgotten treasures" in full_transcript.lower()

    @pytest.mark.integration
    async def test_transcribe_mia_audio_48khz(self, stt, mia_audio_48khz):
        """Test transcription of 48kHz audio (should be resampled)."""
        # Warm up the model
        await stt.warmup()
        
        # Create session to collect transcripts and errors
        session = STTSession(stt)
        
        # Process the audio
        await stt.process_audio(mia_audio_48khz)
        
        # Wait for result
        await session.wait_for_result(timeout=60.0)
        assert not session.errors
        
        # Verify transcript
        full_transcript = session.get_full_transcript()
        assert len(full_transcript) > 0
        assert "forgotten treasures" in full_transcript.lower()

    @pytest.mark.integration
    async def test_empty_audio(self, stt, silence_2s_48khz):
        """Test that empty/silent audio doesn't crash."""
        # Warm up the model
        await stt.warmup()
        
        # Create session to collect transcripts and errors
        session = STTSession(stt)
        
        # Process silent audio
        await stt.process_audio(silence_2s_48khz)
        
        # Wait a bit - should not error but may not produce transcript
        await asyncio.sleep(2.0)
        
        # Should not have errors
        assert not session.errors

