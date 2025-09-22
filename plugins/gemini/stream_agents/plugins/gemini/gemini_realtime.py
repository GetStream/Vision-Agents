import asyncio
import io
import logging
from typing import Optional, List
from aiortc import MediaStreamTrack
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from google import genai
from google.genai.live import AsyncSession
from google.genai.types import LiveConnectConfigDict, Modality, SpeechConfigDict, VoiceConfigDict, \
    PrebuiltVoiceConfigDict, AudioTranscriptionConfigDict, RealtimeInputConfigDict, TurnCoverage, \
    ContextWindowCompressionConfigDict, SlidingWindowDict, HttpOptions, LiveServerMessage, Blob, Part, \
    SessionResumptionConfig

from stream_agents.core.edge.types import Participant, PcmData
from stream_agents.core.llm import realtime
from stream_agents.core.processors import BaseProcessor
import av

from stream_agents.plugins.gemini.video_forwarder import VideoForwarder

from PIL import Image

logger = logging.getLogger(__name__)


"""
TODO:
- Fully document this file
- mcp & functions
- chat/transcription integration
"""

DEFAULT_MODEL = "gemini-2.5-flash-exp-native-audio-thinking-dialog"


class Realtime(realtime.Realtime):
    """

    Audio data in the Live API is always raw, little-endian, 16-bit PCM. Audio output always uses a sample rate of 24kHz.
    Input audio is natively 16kHz, but the Live API will resample if needed
    """
    model : str
    session_resumption_id: Optional[str] = None
    config: LiveConnectConfigDict

    def __init__(self, model: str=DEFAULT_MODEL, config: Optional[LiveConnectConfigDict]=None, http_options: Optional[HttpOptions] = None, client: Optional[genai.Client] = None, api_key: Optional[str] = None ) -> None:
        super().__init__()
        self.model = model
        if http_options is None:
            http_options = HttpOptions(api_version="v1alpha")

        if client is None:  
            if api_key:
                client = genai.Client(api_key=api_key, http_options=http_options)
            else:
                client = genai.Client(http_options=http_options)

        self.client = client
        self.config: LiveConnectConfigDict = self._create_config(config)
        self.logger = logging.getLogger(__name__)
        # Gemini generates at 24k. webrtc automatically translates it to 48khz
        self.output_track = AudioStreamTrack(
            framerate=24000, stereo=False, format="s16"
        )
        self._video_forwarder: Optional[VideoForwarder] = None
        self._session_context = None
        self._session: Optional[AsyncSession] = None
        self._receive_task = None

    def _get_config(self) -> LiveConnectConfigDict:
        config = self.config.copy()
        # resume if we have a session resumption id/handle
        if self.session_resumption_id:
            config["session_resumption"] = SessionResumptionConfig(handle=self.session_resumption_id)
        # set the instructions
        # TODO: potentially we can share the markdown as files/parts.. might do better TBD
        config["system_instruction"] = self._build_enhanced_instructions()
        return config

    async def simple_response(self, text: str, processors: Optional[List[BaseProcessor]] = None,
                              participant: Participant = None):
        """
        Simplify and send the text to send client content
        """
        self.logger.info("Simple response")
        await self.send_realtime_input(text=text)
        self.logger.info("Simple response completed")

    async def simple_audio_response(self, pcm: PcmData):
        self.logger.debug(f"Sending audio to gemini: {pcm.duration}")

        try:
            # Build blob and send directly
            audio_bytes = pcm.samples.tobytes()
            mime = f"audio/pcm;rate={pcm.sample_rate}"
            blob = Blob(data=audio_bytes, mime_type=mime)

            await self._session.send_realtime_input(audio=blob)
        except Exception as e:
            logging.error(e)


    async def send_realtime_input(self, *args, **kwargs):
        """
        Wrap the native send_realtime_input
        """
        await self._session.send_realtime_input(
            *args, **kwargs
        )

    async def send_client_content(self, *args, **kwargs):
        """
        Wrap the native send client content
        """
        try:
            await self._session.send_client_content(
                *args, **kwargs
            )
        except Exception as e:
            # reconnect here in some cases
            self.logger.error(e)
            is_temp = self._is_temporary_error(e)
            if is_temp:
                await self._reconnect()
            else:
                raise

    async def connect(self):
        """
        Connect to Gemini's websocket
        """
        self.logger.info("Connecting to Realtime, config set to %s", self.config)

        # use resumption id here

        self._session_context = self.client.aio.live.connect(model=self.model, config=self._get_config())
        self._session = await self._session_context.__aenter__()
        self.logger.info("Connected to session %s", self._session)

        # Start the receive loop task
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _reconnect(self):
        await self.connect()

    async def _receive_loop(self):
        self.logger.info("_receive_loop started")
        try:
            while True:
                async for response in self._session.receive():
                    response: LiveServerMessage = response

                    is_input_transcript = response and response.server_content and response.server_content.input_transcription
                    is_output_transcript = response and response.server_content and response.server_content.output_transcription
                    is_response = response and response.server_content and response.server_content.model_turn
                    is_interrupt = response and response.server_content and response.server_content.interrupted
                    is_turn_complete = response and response.server_content and response.server_content.turn_complete
                    is_generation_complete = response and response.server_content and response.server_content.generation_complete

                    if is_input_transcript:
                        # TODO: what to do with this? check with Tommaso

                        self.logger.info("input: %s", response.server_content.input_transcription.text)
                    elif is_output_transcript:
                        # TODO: what to do with this?

                        self.logger.info("output: %s", response.server_content.output_transcription.text)
                    elif is_interrupt:
                        self.logger.info("interrupted: %s", response.server_content.interrupted)
                    elif is_response:
                        # Store the resumption id so we can resume a broken connection
                        if response.session_resumption_update:
                            update = response.session_resumption_update
                            if update.resumable and update.new_handle:
                                self.session_resumption_id = update.new_handle

                        parts = response.server_content.model_turn.parts

                        for part in parts:
                            part: Part = part
                            if part.text:
                                if part.thought:
                                    self.logger.info("Gemini thought %s", part.text)
                                else:
                                    print("text", response.text)
                            elif part.inline_data:
                                data = part.inline_data.data
                                # Convert bytes to PcmData at 24kHz (Gemini's output rate)
                                pcm_data = PcmData.from_bytes(data, sample_rate=24000, format="s16")
                                # Resample from 24kHz to 48kHz for WebRTC
                                resampled_pcm = pcm_data.resample(target_sample_rate=48000)
                                # TODO: update to new event syntax
                                # self.emit("audio", resampled_pcm.samples.tobytes()) # audio event is resampled to 48khz
                                await self.output_track.write(data) # original 24khz here
                            else:
                                print("text", response.text)
                    elif is_turn_complete:
                        self.logger.info("is_turn_complete complete")
                    elif is_generation_complete:
                        self.logger.info("is_generation_complete complete")
                    else:
                        self.logger.warning("Unrecognized event structure for gemini %s", response)

        except asyncio.CancelledError:
            self.logger.info("_receive_loop cancelled")
            raise
        except Exception as e:
            # reconnect here for some errors
            self.logger.error(f"_receive_loop error: {e}")
            is_temp = self._is_temporary_error(e)
            if is_temp:
                await self._reconnect()
            else:
                raise
        finally:
            self.logger.info("_receive_loop ended")

    def _is_temporary_error(self, e: Exception):
        should_reconnect = False
        return should_reconnect

    async def _close_impl(self):
        if hasattr(self, '_receive_task') and self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, '_session_context') and self._session_context:
            # Properly close the session using the context manager's __aexit__
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception as e:
                self.logger.warning(f"Error closing session: {e}")
            self._session_context = None
            self._session = None

    @classmethod
    def _frame_to_png_bytes(cls, frame) -> bytes:
        """Convert a video frame to PNG bytes."""
        if Image is None:
            logger.warning("PIL Image not available, cannot convert frame to PNG")
            return b""
        
        try:
            if hasattr(frame, "to_image"):
                img = frame.to_image()
            else:
                arr = frame.to_ndarray(format="rgb24")
                img = Image.fromarray(arr)
            
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error converting frame to PNG: {e}")
            return b""

    async def _watch_video_track(self, input_track: MediaStreamTrack, fps: int = 1) -> None:
        """Start sending video frames to Gemini using VideoForwarder."""
        if self._video_forwarder is not None:
            self.logger.warning("Video sender already running, stopping previous one")
            await self._stop_watching_video_track()
        
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

    async def _stop_watching_video_track(self) -> None:
        """Stop the video sender."""
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            self.logger.info("Stopped video sender")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        if not frame:
            return
        """Send a video frame to Gemini as a PNG blob."""
        if not hasattr(self, '_session') or self._session is None:
            self.logger.warning("No active session, cannot send video frame")
            return
        
        try:
            png_bytes = self.__class__._frame_to_png_bytes(frame)
            blob = Blob(data=png_bytes, mime_type="image/png")
            await self._session.send_realtime_input(media=blob)
        except Exception as e:
            self.logger.error(f"Error sending video frame: {e}")

    def _create_config(self, config: Optional[LiveConnectConfigDict]=None) -> LiveConnectConfigDict:
        default_config = LiveConnectConfigDict(
            response_modalities=[Modality.AUDIO],
            input_audio_transcription=AudioTranscriptionConfigDict(),
            output_audio_transcription=AudioTranscriptionConfigDict(),
            speech_config=SpeechConfigDict(
                voice_config=VoiceConfigDict(
                    prebuilt_voice_config=PrebuiltVoiceConfigDict(voice_name="Puck")
                ),
                language_code="en-US",
            ),
            realtime_input_config=RealtimeInputConfigDict(
                turn_coverage=TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY
            ),
            context_window_compression=ContextWindowCompressionConfigDict(
                trigger_tokens=25600,
                sliding_window=SlidingWindowDict(target_tokens=12800),
            ),
        )
        if config is not None:
            for k, v in config.items():
                default_config[k] = v
        return default_config




