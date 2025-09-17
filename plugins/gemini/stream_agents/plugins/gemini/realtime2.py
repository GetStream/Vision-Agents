import asyncio
import logging
from typing import Optional, List

import numpy as np
from dotenv import load_dotenv
from getstream.audio import resample_audio
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from google import genai
from google.genai.types import LiveConnectConfigDict, Modality, SpeechConfigDict, VoiceConfigDict, \
    PrebuiltVoiceConfigDict, AudioTranscriptionConfigDict, RealtimeInputConfigDict, TurnCoverage, \
    ContextWindowCompressionConfigDict, SlidingWindowDict, HttpOptions, LiveServerMessage, Blob, Part

from stream_agents.core.edge.types import Participant
from stream_agents.core.llm import realtime
from stream_agents.core.processors import BaseProcessor

"""
TODO:
- Why 1000 WS closed? that we have to fix first
- How to forward video to it

- fix init to have all features
- session resumption should work
- mcp & functions
- audio conversion to a utility
- evaluate if we need a task group
- at mention support





"""

DEFAULT_MODEL = "gemini-2.5-flash-exp-native-audio-thinking-dialog"
DEFAULT_MODEL = "gemini-live-2.5-flash-preview"

class Realtime2(realtime.Realtime):
    """

    Audio data in the Live API is always raw, little-endian, 16-bit PCM. Audio output always uses a sample rate of 24kHz. Input audio is natively 16kHz, but the Live API will resample if needed
    """

    async def send_audio_pcm(self, pcm: PcmData, target_rate: int = 24000):
        return
        try:
            #self.logger.info(f"Sending audio pcm: {pcm}")
            # TODO: do we need to send over empty audio? seems like we maybe don't?
            # TODO: why is target rate specified here...
            # Convert to numpy int16 array
            if isinstance(pcm.samples, (bytes, bytearray)):
                # Interpret as int16 little-endian
                audio_array = np.frombuffer(pcm.samples, dtype=np.int16)
            else:
                audio_array = np.asarray(pcm.samples)
                if audio_array.dtype != np.int16:
                    audio_array = audio_array.astype(np.int16)

            # Resample if needed
            if pcm.sample_rate != target_rate:
                audio_array = resample_audio(
                    audio_array, pcm.sample_rate, target_rate
                ).astype(np.int16)

            # Activity detection with a small hysteresis to avoid flapping
            energy = float(np.mean(np.abs(audio_array)))
            is_active = energy > float(1000)

            # Build blob and send directly
            audio_bytes = audio_array.tobytes()
            mime = f"audio/pcm;rate={target_rate}"
            blob = Blob(data=audio_bytes, mime_type=mime)

            await self._session.send_realtime_input(audio=blob)
        except Exception as e:
            logging.error(e)

    async def send_text(self, text: str):
        pass

    async def _close_impl(self):
        return
        self._session.close()

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None,
                              participant: Participant = None):

        self.logger.info("Simple response")
        await self._session.send_client_content(
            turns={"role": "user", "parts": [{"text": text}]}, turn_complete=True
        )

    def __init__(self, model: str =DEFAULT_MODEL, config: Optional[LiveConnectConfigDict]=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        #http_options=HttpOptions(api_version="v1alpha")
        self.client = genai.Client()
        self.config = self._create_config(config)
        self.logger = logging.getLogger(__name__)
        # TODO: this is wrong since we need 48khz for webrtc
        self.output_track = AudioStreamTrack(
            framerate=24000, stereo=False, format="s16"
        )


    def _create_config(self, config: Optional[LiveConnectConfigDict]=None) -> LiveConnectConfigDict:
        default_config = LiveConnectConfigDict(
            response_modalities=[Modality.AUDIO],
            #input_audio_transcription=AudioTranscriptionConfigDict(),
            #output_audio_transcription=AudioTranscriptionConfigDict(),
            speech_config=SpeechConfigDict(
                voice_config=VoiceConfigDict(
                    prebuilt_voice_config=PrebuiltVoiceConfigDict(voice_name="Puck")
                ),
                language_code="en-US",
            ),
            # realtime_input_config=RealtimeInputConfigDict(
            #     turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
            # ),
            # context_window_compression=ContextWindowCompressionConfigDict(
            #     trigger_tokens=25600,
            #     sliding_window=SlidingWindowDict(target_tokens=12800),
            # ),
        )
        if config is not None:
            for k, v in config.items():
                default_config[k] = v
        return default_config



    async def connect(self):
        self.logger.info("Connecting to Realtime, config set to %s", self.config)
        
        print("started")
        
        async def _receive_loop():
            self.logger.info("_receive_loop started")
            async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
                self._session = session
                self.logger.info("_receive_loop connected to session %s", self._session)

                async for response in session.receive():
                    self.logger.info("_receive_loop received response")

                    response: LiveServerMessage = response
                    # skip empty response
                    empty = not response or not response.server_content or not response.server_content.model_turn
                    if empty:
                        self.logger.warning("Empty response received from gemini Realtime %s", response)
                        continue

                    parts = response.server_content.model_turn.parts
                    for part in parts:
                        part : Part = part
                        if part.text:
                            if part.thought:
                                self.logger.info("Got thought %s", part.text)
                            else:
                                print("text", response.text)
                        elif part.inline_data:
                            data = part.inline_data.data
                            self.logger.info("Sending out audio %d %s", len(data), part.inline_data.mime_type)
                            self.emit("audio", data)
                            await self.output_track.write(data)
                        else:
                            print("text", response.text)

        self._audio_receiver_task = asyncio.create_task(_receive_loop())
        self._audio_receiver_task.add_done_callback(lambda t: print(f"Task (_receive_loop) error: {t.exception()}"))
