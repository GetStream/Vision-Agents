import asyncio
import io
import logging
from typing import Optional, List, Callable, Any

import numpy as np
from aiortc import MediaStreamTrack, VideoStreamTrack
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
import av

from stream_agents.plugins.gemini.queue import LatestNQueue
from stream_agents.plugins.gemini.video_forwarder import VideoForwarder

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

"""
TODO:
- Study how to run tasks/ when to run tasks etc (fix it for audio next)
- How to forward video to it (testing)

- at mention support (for docs)
- session resumption should work
- mcp & functions
- evaluate if we need a task group (maybe)
"""

DEFAULT_MODEL = "gemini-2.5-flash-exp-native-audio-thinking-dialog"
DEFAULT_MODEL = "gemini-live-2.5-flash-preview"


class Realtime2(realtime.Realtime):
    """

    Audio data in the Live API is always raw, little-endian, 16-bit PCM. Audio output always uses a sample rate of 24kHz.
    Input audio is natively 16kHz, but the Live API will resample if needed
    """

    def __init__(self, model: str=DEFAULT_MODEL, config: Optional[LiveConnectConfigDict]=None, http_options: Optional[HttpOptions] = None, client: Optional[genai.Client] = None ) -> None:
        super().__init__()
        self.model = model
        if http_options is None:
            http_options = HttpOptions(api_version="v1alpha")

        if client is None:
            client = genai.Client(http_options=http_options)

        self.config = self._create_config(config)
        self.logger = logging.getLogger(__name__)
        self.output_track = AudioStreamTrack(
            framerate=24000, stereo=False, format="s16"
        )
        self._video_forwarder: Optional[VideoForwarder] = None

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None,
                              participant: Participant = None):
        """
        Simplify and send the text to send client content
        """
        self.logger.info("Simple response")
        await self.send_client_content(
            turns={"role": "user", "parts": [{"text": text}]}, turn_complete=True
        )


    async def send_client_content(self, *args, **kwargs):
        """
        Wrap the native send client content
        """
        await self._session.send_client_content(
            *args, **kwargs
        )

    async def connect(self):
        """
        Connect to Gemini's websocket
        """
        self.logger.info("Connecting to Realtime, config set to %s", self.config)

        print("started")

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        self.logger.info("_receive_loop started")
        async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
            self._session = session
            self.logger.info("_receive_loop connected to session %s", self._session)

            while True:
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
                        part: Part = part
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



    async def send_audio_pcm(self, pcm: PcmData, target_rate: int = 24000):
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


    async def _close_impl(self):
        self._session.close()





    def _frame_to_png_bytes(self, frame: Any) -> bytes:
        """Convert a video frame to PNG bytes."""
        if Image is None:
            self.logger.warning("PIL Image not available, cannot convert frame to PNG")
            return b""
        
        try:
            if hasattr(frame, "to_image"):
                img = frame.to_image()  # type: ignore[attr-defined]
            else:
                arr = frame.to_ndarray(format="rgb24")  # type: ignore[attr-defined]
                img = Image.fromarray(arr)
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            self.logger.error(f"Error converting frame to PNG: {e}")
            return b""

    async def start_video_sender(self, input_track: MediaStreamTrack, fps: int = 1) -> None:
        """Start sending video frames to Gemini using VideoForwarder."""
        if self._video_forwarder is not None:
            self.logger.warning("Video sender already running, stopping previous one")
            await self.stop_video_sender()
        
        # Create VideoForwarder with the input track
        self._video_forwarder = VideoForwarder(
            input_track,  # type: ignore[arg-type]
            max_buffer=5,
            fps=float(fps)
        )
        
        # Start the forwarder
        await self._video_forwarder.start()
        
        # Start the callback consumer that sends frames to Gemini
        await self._video_forwarder.start_event_consumer(self._send_video_frame)
        
        self.logger.info(f"Started video sender with {fps} FPS")

    async def stop_video_sender(self) -> None:
        """Stop the video sender."""
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            self.logger.info("Stopped video sender")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """Send a video frame to Gemini as a PNG blob."""
        if not hasattr(self, '_session') or self._session is None:
            self.logger.warning("No active session, cannot send video frame")
            return
        
        try:
            png_bytes = self._frame_to_png_bytes(frame)
            if png_bytes:
                blob = Blob(data=png_bytes, mime_type="image/png")
                await self._session.send_realtime_input(media=blob)
                self.logger.debug(f"Sent video frame ({len(png_bytes)} bytes)")
        except Exception as e:
            self.logger.error(f"Error sending video frame: {e}")

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




