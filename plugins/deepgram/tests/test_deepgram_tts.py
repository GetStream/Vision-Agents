import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData

from vision_agents.core.tts.manual_test import manual_tts_to_wav
from vision_agents.core.tts.testing import TTSSession
from vision_agents.plugins import deepgram
from vision_agents.plugins.deepgram.tts import (
    SpeakV1Cleared,
    SpeakV1Flushed,
    SpeakV1Warning,
)

load_dotenv()


def _pcm_bytes(n_samples: int = 160) -> bytes:
    """Generate valid linear16 PCM bytes."""
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_mock_socket(messages: list[object]) -> AsyncMock:
    """Create a mock AsyncV1SocketClient that yields the given messages."""
    socket = AsyncMock()
    socket.send_text = AsyncMock()
    socket.send_flush = AsyncMock()
    socket.send_clear = AsyncMock()
    socket.send_close = AsyncMock()

    async def _aiter(self_):
        for msg in messages:
            yield msg

    socket.__aiter__ = _aiter
    return socket


class _MockConnectCtx:
    def __init__(self, socket: AsyncMock):
        self.socket = socket
        self.exit_called = False

    async def __aenter__(self):
        return self.socket

    async def __aexit__(self, *args):
        self.exit_called = True


class TestDeepgramTTS:
    """Integration tests for Deepgram TTS."""

    @pytest.fixture
    async def tts(self) -> deepgram.TTS:
        return deepgram.TTS()

    @pytest.mark.integration
    async def test_deepgram_tts_convert_text_to_audio(self, tts: deepgram.TTS):
        tts.set_output_format(sample_rate=16000, channels=1)
        session = TTSSession(tts)
        text = "Hello from Deepgram."

        await tts.send(text)
        await session.wait_for_result(timeout=15.0)

        assert not session.errors
        assert len(session.speeches) > 0

    @pytest.mark.integration
    async def test_deepgram_tts_convert_text_to_audio_manual_test(
        self, tts: deepgram.TTS
    ):
        await manual_tts_to_wav(tts, sample_rate=48000, channels=2)


class TestDeepgramTTSWebSocket:
    """Unit tests for the websocket-based TTS implementation."""

    async def test_stream_audio_yields_chunks(self):
        mock_socket = _make_mock_socket(
            [
                _pcm_bytes(800),
                _pcm_bytes(800),
                SpeakV1Flushed(type="Flushed", sequence_id=1),
            ]
        )
        mock_ctx = _MockConnectCtx(mock_socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        chunks = [chunk async for chunk in result]

        # silence head + 2 audio chunks + silence tail/pause
        assert len(chunks) == 4
        assert all(isinstance(c, PcmData) for c in chunks)
        mock_socket.send_text.assert_called_once()
        mock_socket.send_flush.assert_called_once()

    async def test_stream_audio_ignores_cleared(self):
        mock_socket = _make_mock_socket(
            [
                _pcm_bytes(800),
                SpeakV1Cleared(type="Cleared", sequence_id=1),
                _pcm_bytes(800),
                SpeakV1Flushed(type="Flushed", sequence_id=2),
            ]
        )
        mock_ctx = _MockConnectCtx(mock_socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        chunks = [chunk async for chunk in result]

        # silence head + 2 audio chunks + silence tail/pause
        assert len(chunks) == 4

    async def test_stale_cleared_before_audio(self):
        mock_socket = _make_mock_socket(
            [
                SpeakV1Cleared(type="Cleared", sequence_id=0),
                _pcm_bytes(800),
                SpeakV1Flushed(type="Flushed", sequence_id=1),
            ]
        )
        mock_ctx = _MockConnectCtx(mock_socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        chunks = [chunk async for chunk in result]

        # silence head + audio chunk + silence tail/pause
        assert len(chunks) == 3
        assert chunks[1].samples.size == 800

    async def test_stream_audio_skips_warnings(self):
        mock_socket = _make_mock_socket(
            [
                _pcm_bytes(800),
                SpeakV1Warning(type="Warning", description="test", code="W001"),
                _pcm_bytes(800),
                SpeakV1Flushed(type="Flushed", sequence_id=1),
            ]
        )
        mock_ctx = _MockConnectCtx(mock_socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        chunks = [chunk async for chunk in result]

        # silence head + 2 audio chunks + silence tail/pause
        assert len(chunks) == 4

    async def test_stop_audio_sends_clear(self):
        mock_socket = _make_mock_socket([])
        tts = deepgram.TTS()
        tts._socket = mock_socket

        await tts.stop_audio()

        mock_socket.send_clear.assert_called_once()
        assert tts._needs_drain
        assert tts._stop_event.is_set()

    async def test_stop_audio_noop_when_not_connected(self):
        tts = deepgram.TTS()
        await tts.stop_audio()

    async def test_stop_event_terminates_receive(self):
        socket = AsyncMock()
        socket.send_text = AsyncMock()
        socket.send_flush = AsyncMock()

        async def _aiter(self_):
            yield _pcm_bytes(800)
            await asyncio.sleep(10)

        socket.__aiter__ = _aiter
        mock_ctx = _MockConnectCtx(socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        tts._stop_event.set()

        chunks = []
        async for chunk in result:
            chunks.append(chunk)
            if len(chunks) > 10:
                break
        assert len(chunks) >= 1

    async def test_close_tears_down_connection(self):
        mock_socket = _make_mock_socket([])
        mock_ctx = _MockConnectCtx(mock_socket)

        tts = deepgram.TTS()
        tts._socket = mock_socket
        tts._connection_ctx = mock_ctx

        await tts.close()

        mock_socket.send_close.assert_called_once()
        assert mock_ctx.exit_called
        assert tts._socket is None

    async def test_connection_reused_across_calls(self):
        call_count = 0
        batches = [
            [_pcm_bytes(800), SpeakV1Flushed(type="Flushed", sequence_id=1)],
            [_pcm_bytes(800), SpeakV1Flushed(type="Flushed", sequence_id=2)],
        ]

        socket = AsyncMock()
        socket.send_text = AsyncMock()
        socket.send_flush = AsyncMock()

        async def _aiter(self_):
            nonlocal call_count
            for msg in batches[call_count]:
                yield msg
            call_count += 1

        socket.__aiter__ = _aiter
        mock_ctx = _MockConnectCtx(socket)

        tts = deepgram.TTS()
        tts.client = MagicMock()
        tts.client.speak.v1.connect.return_value = mock_ctx

        result = await tts.stream_audio("Hello")
        _ = [chunk async for chunk in result]
        assert tts.client.speak.v1.connect.call_count == 1

        result = await tts.stream_audio("World")
        _ = [chunk async for chunk in result]
        assert tts.client.speak.v1.connect.call_count == 1
