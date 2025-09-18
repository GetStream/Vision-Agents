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

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

"""
TODO:
- Study how to run tasks/ when to run tasks etc
- How to forward video to it

- fix init to have all features
- session resumption should work
- mcp & functions
- audio conversion to a utility
- evaluate if we need a task group
- at mention support





"""


class VideoForwarder:
    """
    Pulls frames from `input_track` into a latest-N buffer.
    Consumers can:
      - call `await next_frame()` (pull model), OR
      - run `start_event_consumer(on_frame)` (push model via callback).
    `fps` limits how often frames are forwarded to consumers (coalescing to newest).
    """
    def __init__(self, input_track: VideoStreamTrack, *, max_buffer: int = 10, fps: Optional[float] = None):
        self.input_track = input_track
        self.queue: LatestNQueue[av.VideoFrame] = LatestNQueue(maxlen=max_buffer)
        self.fps = fps  # None = unlimited, else forward at ~fps
        self._tasks: set[asyncio.Task] = set()
        self._stopped = asyncio.Event()

    # ---------- lifecycle ----------
    async def start(self) -> None:
        self._stopped.clear()
        self._tasks.add(asyncio.create_task(self._producer()))

    async def stop(self) -> None:
        self._stopped.set()
        for t in list(self._tasks):
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        # drain queue
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

    # ---------- producer (fills latest-N buffer) ----------
    async def _producer(self):
        try:
            while not self._stopped.is_set():
                frame = await self.input_track.recv()
                await self.queue.put_latest(frame)
        except asyncio.CancelledError:
            raise
        except Exception:
            # optional: log
            pass

    # ---------- consumer API (pull one frame; coalesce backlog to newest) ----------
    async def next_frame(self, *, timeout: Optional[float] = None) -> av.VideoFrame:
        """
        Returns the newest available frame. If there's backlog, older frames
        are drained so you get the latest (low latency).
        """
        if timeout is None:
            frame = await self.queue.get()
        else:
            async with asyncio.timeout(timeout):
                frame = await self.queue.get()

        # drain to newest
        while True:
            try:
                newer = self.queue.get_nowait()
                frame = newer
            except asyncio.QueueEmpty:
                break
        return frame

    # ---------- push model (broadcast via callback) ----------
    async def start_event_consumer(
        self,
        on_frame: Callable[[av.VideoFrame], Any],  # async or sync
    ) -> None:
        """
        Starts a task that calls `on_frame(latest_frame)` at ~fps.
        If fps is None, it forwards as fast as frames arrive (still coalescing).
        """
        async def _consumer():
            loop = asyncio.get_running_loop()
            min_interval = (1.0 / self.fps) if (self.fps and self.fps > 0) else 0.0
            last_ts = 0.0
            is_coro = asyncio.iscoroutinefunction(on_frame)
            try:
                while not self._stopped.is_set():
                    # Wait for at least one frame
                    frame = await self.next_frame()
                    # Throttle to fps (if set)
                    if min_interval > 0.0:
                        now = loop.time()
                        elapsed = now - last_ts
                        if elapsed < min_interval:
                            # coalesce: keep draining to newest until it's time
                            await asyncio.sleep(min_interval - elapsed)
                        last_ts = loop.time()
                    # Call handler
                    if is_coro:
                        await on_frame(frame)  # type: ignore[arg-type]
                    else:
                        on_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                # optional: log
                pass

        self._tasks.add(asyncio.create_task(_consumer()))


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
        self._video_forwarder: Optional[VideoForwarder] = None

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
