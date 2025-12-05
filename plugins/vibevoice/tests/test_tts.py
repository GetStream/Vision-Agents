import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from getstream.video.rtc.track_util import PcmData

from vision_agents.plugins.vibevoice.tts import VibeVoiceTTS


class TestVibeVoiceTTS(unittest.IsolatedAsyncioTestCase):
    async def test_init_defaults(self):
        tts = VibeVoiceTTS()
        self.assertEqual(tts.mode, "local")
        self.assertEqual(tts.model_id, "microsoft/VibeVoice-Realtime-0.5B")

    async def test_init_remote(self):
        tts = VibeVoiceTTS(base_url="ws://test:3000")
        self.assertEqual(tts.mode, "remote")
        self.assertEqual(tts.base_url, "ws://test:3000")

    @patch("vision_agents.plugins.vibevoice.tts.websockets.connect")
    async def test_stream_audio_remote(self, mock_connect):
        # Setup mock websocket
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws

        # Mock messages
        # 1. binary audio chunk (pcm s16)
        # 2. log message (json)
        audio_bytes = np.zeros(1600, dtype=np.int16).tobytes()  # 1600 samples
        mock_ws.__aiter__.return_value = [audio_bytes, '{"event": "log"}']

        tts = VibeVoiceTTS(base_url="ws://test:3000")

        chunks = []
        async for chunk in tts.stream_audio("Hello world"):
            chunks.append(chunk)

        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], PcmData)
        self.assertEqual(chunks[0].samples.tobytes(), audio_bytes)

        # Verify connection args
        mock_connect.assert_called_once()
        url = mock_connect.call_args[0][0]
        self.assertIn("ws://test:3000/stream", url)
        self.assertIn("text=Hello+world", url)

    @patch("vision_agents.plugins.vibevoice.tts.HAS_VIBEVOICE", True)
    @patch("vision_agents.plugins.vibevoice.tts.snapshot_download")
    @patch("vision_agents.plugins.vibevoice.tts.VibeVoiceStreamingProcessor")
    @patch(
        "vision_agents.plugins.vibevoice.tts.VibeVoiceStreamingForConditionalGenerationInference"
    )
    @patch("vision_agents.plugins.vibevoice.tts.torch")
    async def test_warmup_local(
        self, mock_torch, mock_model_cls, mock_proc_cls, mock_download
    ):
        mock_download.return_value = "/tmp/model"

        tts = VibeVoiceTTS()

        # Mock model instance
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock torch device
        mock_torch.device.return_value = "cpu"
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False

        tts.warmup()

        mock_download.assert_called_with(repo_id="microsoft/VibeVoice-Realtime-0.5B")
        mock_proc_cls.from_pretrained.assert_called_with("/tmp/model")
        mock_model_cls.from_pretrained.assert_called()
        self.assertIsNotNone(tts.model)

    @patch("vision_agents.plugins.vibevoice.tts.HAS_VIBEVOICE", True)
    @patch("vision_agents.plugins.vibevoice.tts.VibeVoiceTTS.warmup")
    @patch("vision_agents.plugins.vibevoice.tts.AudioStreamer")
    @patch("threading.Thread")
    @patch(
        "vision_agents.plugins.vibevoice.tts.os.path.exists"
    )  # Mock file exists for cleanup
    async def test_stream_audio_local(
        self, mock_exists, mock_thread, mock_streamer_cls, mock_warmup
    ):
        tts = VibeVoiceTTS()
        tts.model = MagicMock()
        tts.processor = MagicMock()
        tts.voice_cache = {"en-WHTest_man": MagicMock()}  # Mock cache to avoid load

        # Mock streamer iteration
        mock_streamer = MagicMock()
        mock_streamer_cls.return_value = mock_streamer

        mock_exists.return_value = True

        # Mock thread start to populate queue
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        def start_side_effect():
            # Extract args passed to Thread constructor
            call_args = mock_thread.call_args
            if call_args:
                _, kwargs = call_args
                # args: (text, queue_out, loop)
                queue_out = kwargs["args"][1]
                loop = kwargs["args"][2]

                # Verify passed args
                self.assertEqual(kwargs["args"][0], "Test")

                # Put dummy data
                data = PcmData.from_bytes(
                    b"\0\0", sample_rate=24000, channels=1, format="s16"
                )
                loop.call_soon_threadsafe(queue_out.put_nowait, data)
                loop.call_soon_threadsafe(queue_out.put_nowait, None)

        mock_thread_instance.start.side_effect = start_side_effect

        chunks = []
        async for chunk in tts.stream_audio("Test"):
            chunks.append(chunk)

        mock_thread.assert_called()
        self.assertTrue(mock_thread_instance.start.called)
        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], PcmData)
