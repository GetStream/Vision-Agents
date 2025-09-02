"""Tests for the FalWizperSTT plugin."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from getstream.plugins import FalWizperSTT
from getstream.video.rtc.track_util import PcmData


@pytest.fixture
def stt():
    """Provides a FalWizperSTT instance with a mocked fal_client."""
    with patch(
        "getstream.plugins.fal.stt.stt.fal_client.AsyncClient"
    ) as mock_fal_client:
        stt_instance = FalWizperSTT()
        stt_instance._fal_client = mock_fal_client.return_value
        yield stt_instance


class TestFalWizperSTT:
    """Test suite for the FalWizperSTT class."""

    def test_init(self):
        """Test that the __init__ method sets attributes correctly."""
        stt = FalWizperSTT(task="translate", target_language="es", sample_rate=16000)
        assert stt.task == "translate"
        assert stt.target_language == "es"
        assert stt.sample_rate == 16000
        assert not stt._is_closed

    def test_pcm_to_wav_bytes(self, stt):
        """Test the conversion of PCM data to WAV bytes."""
        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 63).astype(np.int16)
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )
        wav_bytes = stt._pcm_to_wav_bytes(pcm_data)
        assert wav_bytes.startswith(b"RIFF")
        assert b"WAVE" in wav_bytes

    @pytest.mark.asyncio
    async def test_process_audio_impl_success_transcribe(self, stt):
        """Test successful transcription with a valid response."""
        stt._fal_client.upload_file = AsyncMock(
            return_value="http://mock.url/audio.wav"
        )
        stt._fal_client.subscribe = AsyncMock(
            return_value={"text": " This is a test. ", "chunks": []}
        )

        transcript_handler = AsyncMock()
        stt.on("transcript", transcript_handler)

        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 8191).astype(np.int16)
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )

        with (
            patch(
                "tempfile.NamedTemporaryFile", new_callable=MagicMock
            ) as mock_temp_file,
            patch("os.unlink", new_callable=MagicMock) as mock_unlink,
        ):
            await stt._process_audio_impl(pcm_data, {"user": "test_user"})
            await asyncio.sleep(0)  # Allow event loop to run

            mock_temp_file.assert_called_once_with(suffix=".wav", delete=False)
            stt._fal_client.upload_file.assert_awaited_once()
            stt._fal_client.subscribe.assert_awaited_once()

            subscribe_args = stt._fal_client.subscribe.call_args.kwargs
            assert (
                subscribe_args["arguments"]["audio_url"] == "http://mock.url/audio.wav"
            )
            assert subscribe_args["arguments"]["task"] == "transcribe"
            assert "language" not in subscribe_args["arguments"]

            # Check that the transcript handler was called with an event object
            transcript_handler.assert_called_once()
            call_args = transcript_handler.call_args[0]
            assert len(call_args) == 1
            event = call_args[0]
            assert event.text == "This is a test."
            assert event.user_metadata == {"user": "test_user"}
            mock_unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_audio_impl_success_translate(self):
        """Test successful translation with target language."""
        with patch("getstream.plugins.fal.stt.stt.fal_client.AsyncClient"):
            stt = FalWizperSTT(task="translate", target_language="pt")
            stt._fal_client.upload_file = AsyncMock(
                return_value="http://mock.url/audio.wav"
            )
            stt._fal_client.subscribe = AsyncMock(
                return_value={"text": "This is a test.", "chunks": []}
            )

            samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 32767).astype(
                np.int16
            )
            pcm_data = PcmData(
                samples=samples,
                sample_rate=48000,
                format="s16",
            )
            with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
                await stt._process_audio_impl(pcm_data)

            subscribe_args = stt._fal_client.subscribe.call_args.kwargs
            assert subscribe_args["arguments"]["language"] == "pt"

    @pytest.mark.asyncio
    async def test_process_audio_impl_no_text(self, stt):
        """Test that no transcript is emitted if the API response lacks 'text'."""
        stt._fal_client.upload_file = AsyncMock(
            return_value="http://mock.url/audio.wav"
        )
        stt._fal_client.subscribe = AsyncMock(
            return_value={"chunks": []}
        )  # No 'text' field

        transcript_handler = AsyncMock()
        stt.on("transcript", transcript_handler)

        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 32767).astype(
            np.int16
        )
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            await stt._process_audio_impl(pcm_data)

        transcript_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_audio_impl_empty_text(self, stt):
        """Test that no transcript is emitted for empty or whitespace-only text."""
        stt._fal_client.upload_file = AsyncMock(
            return_value="http://mock.url/audio.wav"
        )
        stt._fal_client.subscribe = AsyncMock(
            return_value={"text": "  ", "chunks": []}
        )  # Empty text

        transcript_handler = AsyncMock()
        stt.on("transcript", transcript_handler)

        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 32767).astype(
            np.int16
        )
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            await stt._process_audio_impl(pcm_data)

        transcript_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_audio_impl_api_error(self, stt):
        """Test that an error event is emitted when the API call fails."""
        stt._fal_client.upload_file = AsyncMock(side_effect=Exception("API Error"))

        error_handler = AsyncMock()
        stt.on("error", error_handler)

        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 32767).astype(
            np.int16
        )
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )

        with patch("tempfile.NamedTemporaryFile"), patch("os.unlink"):
            await stt._process_audio_impl(pcm_data)
            await asyncio.sleep(0)  # Allow event loop to run

        # Check that the error handler was called with an event object
        error_handler.assert_called_once()
        call_args = error_handler.call_args[0]
        assert len(call_args) == 1
        event = call_args[0]
        assert isinstance(event.error, Exception)
        assert str(event.error) == "API Error"

    @pytest.mark.asyncio
    async def test_process_audio_impl_empty_audio(self, stt):
        """Test that no API call is made for empty audio data."""
        pcm_data = PcmData(
            samples=np.array([], dtype=np.int16),
            sample_rate=48000,
            format="s16",
        )
        await stt._process_audio_impl(pcm_data)
        stt._fal_client.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_audio_impl_when_closed(self, stt):
        """Test that audio is ignored if the STT service is closed."""
        await stt.close()
        samples = (np.sin(np.linspace(0, 440 * 2 * np.pi, 480)) * 32767).astype(
            np.int16
        )
        pcm_data = PcmData(
            samples=samples,
            sample_rate=48000,
            format="s16",
        )
        await stt._process_audio_impl(pcm_data)
        stt._fal_client.upload_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_close(self, stt):
        """Test that the close method works correctly and is idempotent."""
        assert not stt._is_closed
        await stt.close()
        assert stt._is_closed
        # Test idempotency
        await stt.close()
        assert stt._is_closed
