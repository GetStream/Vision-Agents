import logging

from getstream.plugins.common import TTS
from elevenlabs.client import AsyncElevenLabs
from getstream.video.rtc.audio_track import AudioStreamTrack
from typing import AsyncIterator, Optional
import os


class ElevenLabsTTS(TTS):
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

    def set_output_track(self, track: AudioStreamTrack) -> None:
        if track.framerate != 16000:
            raise TypeError("Invalid framerate, audio track only supports 16000")
        super().set_output_track(track)

    async def stream_audio(self, text: str, *args, **kwargs) -> AsyncIterator[bytes]:
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
            *args,
            **kwargs,
        )

        return audio_stream

    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        Returns:
            None
        """
        if self.track is not None:
            try:
                await self.track.flush()
                logging.info("🎤 Stopping audio track for TTS")
            except Exception as e:
                logging.error(f"Error flushing audio track: {e}")
        else:
            logging.warning("No audio track to stop")
