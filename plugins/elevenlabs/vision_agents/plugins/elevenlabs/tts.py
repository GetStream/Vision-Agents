import logging
import os
from typing import AsyncIterator, Iterator, Optional

from elevenlabs.client import AsyncElevenLabs
from vision_agents.core import tts
from getstream.video.rtc.track_util import PcmData, AudioFormat

logger = logging.getLogger(__name__)


class TTS(tts.TTS):
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "VR6AewLTigWG4xSOukaG",  # Default ElevenLabs voice
        model_id: str = "eleven_multilingual_v2",
        client: Optional[AsyncElevenLabs] = None,
    ):
        """
        Initialize the ElevenLabs TTS service.

        Args:
            api_key: ElevenLabs API key. If not provided, the ELEVENLABS_API_KEY
                    environment variable will be used automatically.
            voice_id: The voice ID to use for synthesis
            model_id: The model ID to use for synthesis
            client: Optionally pass in your own instance of the ElvenLabs Client.
        """
        super().__init__()

        # elevenlabs sdk does not always load the env correctly (default kwarg)
        if not api_key:
            api_key = os.environ.get("ELEVENLABS_API_KEY")

        self.client = client if client is not None else AsyncElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = "pcm_16000"

    async def stream_audio(
        self, text: str, *_, **__
    ) -> PcmData | Iterator[PcmData] | AsyncIterator[PcmData]:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text: The text to convert to speech

        Returns:
            An async iterator of audio chunks as bytes
        """

        audio_stream = self.client.text_to_speech.stream(
            text=text,
            voice_id=self.voice_id,
            output_format=self.output_format,
            model_id=self.model_id,
            request_options={"chunk_size": 64000},
        )

        return PcmData.from_response(
            audio_stream, sample_rate=16000, channels=1, format=AudioFormat.S16
        )

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        pass
