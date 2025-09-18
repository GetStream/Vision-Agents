"""OpenAI Realtime API with improved video reliability using adaptive video tracks."""

import asyncio
from contextlib import asynccontextmanager
import contextlib
from fractions import Fraction
import json
import time
import os
import logging
import httpx
from typing import Any, Optional

from aiortc import (
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaStreamError, MediaStreamTrack
from aiortc.mediastreams import VideoStreamTrack as _VideoStreamTrack
from av import AudioFrame, VideoFrame
from getstream.audio.utils import resample_audio
from getstream.video.call import Call
import numpy as np
from stream_agents.core.events import RealtimeConnectedEvent, register_global_event
from stream_agents.core.llm import realtime
from stream_agents.core.llm.llm import LLMResponse

logger = logging.getLogger(__name__)


class AdaptiveVideoTrack(_VideoStreamTrack):
    """Adaptive video track that switches between black frames and forwarded video."""

    kind = "video"

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 1, outer_conn=None):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = max(1, int(fps))
        self._outer = outer_conn
        self._last_ts = None

        # Source management
        self._source_track: Optional[MediaStreamTrack] = None
        self._source_lock = asyncio.Lock()
        self._forwarding_enabled = False

        logger.debug(f"AdaptiveVideoTrack initialized: {width}x{height} @ {fps}fps")

    async def set_source_track(self, source: Optional[MediaStreamTrack], fps_limit: Optional[int] = None):
        """Switch video source."""
        async with self._source_lock:
            if fps_limit is not None:
                self._fps = max(1, int(fps_limit))

            old_source = self._source_track
            self._source_track = source
            self._forwarding_enabled = source is not None

            if old_source and old_source != source:
                with contextlib.suppress(Exception):
                    if hasattr(old_source, 'stop'):
                        old_source.stop()

            mode = "forwarding" if self._forwarding_enabled else "black frames"
            logger.info(f"Video track switched to {mode} mode")
            if source:
                logger.debug(f"New source: {source.__class__.__name__}")
            else:
                logger.debug("Source cleared, reverting to black frames")

            # Enable test pattern when no source is available for debugging
            self._use_test_pattern = source is None and os.getenv("DEBUG_VIDEO_PATTERN") == "1"

    async def recv(self):
        """Generate frames from source or black frames."""
        # Frame rate limiting
        interval = 1.0 / float(self._fps)
        now = time.monotonic()
        if self._last_ts is not None:
            remaining = (self._last_ts + interval) - now
            if remaining > 0:
                await asyncio.sleep(remaining)
                now = time.monotonic()
        self._last_ts = now

        # Get frame based on current mode
        async with self._source_lock:
            if self._forwarding_enabled and self._source_track:
                frame = await self._get_latest_source_frame()
                if frame is None:
                    frame = self._generate_black_frame()
                    if not getattr(self, '_logged_fallback', False):
                        logger.debug("No source frame available, using black")
                        self._logged_fallback = True
                else:
                    if not getattr(self, '_logged_success', False):
                        logger.debug(f"Got frame from source {frame.width}x{frame.height}")
                        self._logged_success = True
            else:
                frame = self._generate_black_frame()
                if not getattr(self, '_logged_black_mode', False):
                    logger.debug(f"Black mode: forwarding_enabled={self._forwarding_enabled}")
                    self._logged_black_mode = True

        # Apply scaling if needed
        if frame.width > self._width or frame.height > self._height:
            frame = self._scale_frame(frame)

        # Set timing
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base

        # Handle diagnostics if enabled
        if self._outer and self._outer._video_debug_enabled:
            await self._handle_frame_diagnostics(frame)

        return frame

    async def _get_latest_source_frame(self) -> Optional[VideoFrame]:
        """Get the latest frame from source."""
        if not self._source_track:
            return None

        frame = None
        try:
            # Drain old frames to get the latest
            frames_drained = 0
            while True:
                try:
                    f = await asyncio.wait_for(self._source_track.recv(), timeout=0.001)
                    frame = f  # Keep latest
                    frames_drained += 1
                except asyncio.TimeoutError:
                    break
                except MediaStreamError as e:
                    logger.debug(f"Source video stream ended: {e}")
                    return None

            # If no immediate frame, wait for next
            if frame is None:
                frame = await asyncio.wait_for(self._source_track.recv(), timeout=0.1)
                frames_drained = 1

            if frames_drained > 1:
                logger.debug(f"Drained {frames_drained} frames, using latest")
            elif frames_drained == 1:
                logger.debug(f"Got fresh frame {frame.width}x{frame.height}")

            return frame

        except asyncio.TimeoutError:
            logger.debug("No frame from source track within timeout")
            return None
        except Exception as e:
            logger.error(f"Failed to get source frame: {e}")
            return None

    def _generate_black_frame(self) -> VideoFrame:
        """Generate a black video frame with optional test pattern."""
        if getattr(self, '_use_test_pattern', False):
            # Generate a test pattern for debugging
            frame_array = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            h, w = self._height, self._width
            frame_array[:h//3, :, 0] = 255  # Red top third
            frame_array[h//3:2*h//3, :, 1] = 255  # Green middle third
            frame_array[2*h//3:, :, 2] = 255  # Blue bottom third
            return VideoFrame.from_ndarray(frame_array, format="bgr24")
        else:
            # Standard black frame
            black_array = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            return VideoFrame.from_ndarray(black_array, format="bgr24")

    def _scale_frame(self, frame: VideoFrame) -> VideoFrame:
        """Scale frame to target dimensions."""
        if frame.width <= self._width and frame.height <= self._height:
            return frame

        scale = min(self._width / frame.width, self._height / frame.height)
        new_w = max(1, int(frame.width * scale))
        new_h = max(1, int(frame.height * scale))

        try:
            return frame.reformat(width=new_w, height=new_h)
        except Exception as e:
            logger.warning(f"Frame scaling failed: {e}")
            return frame

    async def _handle_frame_diagnostics(self, frame: VideoFrame):
        """Handle frame diagnostics."""
        if not self._outer:
            return

        try:
            arr = frame.to_ndarray(format="rgb24")
            self._outer._last_frame_rgb = arr
            self._outer._last_frame_ts = time.monotonic()

            if not getattr(self, '_first_logged', False):
                mean = float(arr.mean()) if arr.size else 0.0
                h, w, c = arr.shape
                mode = "forward" if self._forwarding_enabled else "black"
                logger.debug(f"First adaptive frame: {w}x{h} mean={mean:.2f} mode={mode}")
                self._first_logged = True

        except Exception as e:
            logger.debug(f"Frame diagnostics failed: {e}")


class ImprovedRealtimeConnection:
    """WebRTC connection with adaptive video track."""

    def __init__(
            self,
            api_key: str,
            model: str = "gpt-realtime",
            voice: str = "alloy",
            turn_detection: bool = True,
            system_instructions: Optional[str] = None,
            enable_video_input: bool = False,
            video_fps: int = 1,
            target_video_width: int = 1280,
            target_video_height: int = 720,
            video_debug_enabled: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.turn_detection = turn_detection
        self.system_instructions = system_instructions
        self.enable_video_input = enable_video_input
        self._video_fps = max(1, int(video_fps))

        self.pc = RTCPeerConnection()
        self.dc: Optional[RTCDataChannel] = None
        self.session_created_event = asyncio.Event()
        self._audio_callbacks = []
        self._event_callbacks = []
        self._running = False

        # Audio handling
        self._mic_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._max_queue_frames: int = 3
        self._frames_dropped: int = 0

        # HTTP client
        self._http: Optional[httpx.AsyncClient] = None

        # Video track management
        self._video_track: Optional[AdaptiveVideoTrack] = None
        self._video_sender = None

        # Video diagnostics
        self._last_frame_rgb: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0
        self._video_debug_enabled: bool = bool(video_debug_enabled)
        self._target_video_width: int = max(1, int(target_video_width))
        self._target_video_height: int = max(1, int(target_video_height))

        # Connection state
        self._last_conn_state: Optional[str] = None
        self.openai_session_id: Optional[str] = None

    async def __aenter__(self):
        await self._start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stop_session()

    def on_audio(self, callback):
        self._audio_callbacks.append(callback)

    def on_event(self, callback):
        self._event_callbacks.append(callback)

    async def send_audio(self, audio_data: bytes, sample_rate: int = 48000):
        """Send audio to OpenAI via microphone track queue."""
        if not audio_data:
            return

        # Backpressure management
        while self._mic_queue.qsize() >= self._max_queue_frames:
            try:
                self._mic_queue.get_nowait()
                self._frames_dropped += 1
            except asyncio.QueueEmpty:
                break

        self._mic_queue.put_nowait((audio_data, sample_rate))

    async def send_text(self, text: str):
        """Send text message to OpenAI."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self._send_event(event)
        await self._send_event({"type": "response.create"})

    async def _send_event(self, event: dict):
        """Send event through data channel."""
        if not self.dc:
            logger.warning("Data channel not ready")
            return

        try:
            self.dc.send(json.dumps(event))
            logger.debug(f"Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def start_video_sender(self, source_track: MediaStreamTrack, fps: Optional[int] = None):
        """Start video forwarding."""
        try:
            if not self.enable_video_input or not self._video_track:
                raise RuntimeError("Video input not enabled")

            logger.info(f"Starting video sender with source: {source_track.__class__.__name__}")
            await self._video_track.set_source_track(source_track, fps)
            logger.info("Video forwarding started successfully")

        except Exception as e:
            logger.error(f"Failed to start video sender: {e}")
            raise

    async def stop_video_sender(self):
        """Stop video forwarding and revert to black frames."""
        try:
            if self._video_track:
                await self._video_track.set_source_track(None)
                logger.info("Video forwarding stopped")
        except Exception as e:
            logger.error(f"Failed to stop video sender: {e}")

    async def _start_session(self):
        """Start WebRTC session."""
        try:
            if self._http is None:
                self._http = httpx.AsyncClient()

            # Get session token (same as original)
            self.token = await self._get_session_token()
            if not self.token:
                raise RuntimeError("Failed to obtain session token")

            # Set up event handlers (same as original)
            @self.pc.on("track")
            async def on_track(track):
                if track.kind == "audio":
                    logger.info(f"Remote audio track attached {track}")
                    asyncio.create_task(self._process_audio_track(track))

            @self.pc.on("connectionstatechange")
            async def on_conn_state_change():
                self._last_conn_state = self.pc.connectionState
                logger.debug(f"Connection state: {self.pc.connectionState}")

            # Data channel setup (same as original)
            self.dc = self.pc.createDataChannel("oai-events")

            @self.dc.on("open")
            def on_open():
                logger.debug("Data channel opened")
                if not self.session_created_event.is_set():
                    self.session_created_event.set()

            @self.dc.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    asyncio.create_task(self._handle_event(data))
                except json.JSONDecodeError as e:
                    logger.error(f"Message decode failed: {e}")

            # Add microphone audio track
            from aiortc.mediastreams import AudioStreamTrack

            class MicAudioTrack(AudioStreamTrack):
                kind = "audio"

                def __init__(self, queue, sample_rate=48000):
                    super().__init__()
                    self._queue = queue
                    self._sample_rate = sample_rate
                    self._ts = 0
                    self._silence_cache = {}

                async def recv(self):
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                        data, sr = item if isinstance(item, tuple) else (item, self._sample_rate)
                        self._sample_rate = int(sr) if sr else self._sample_rate
                        arr = np.frombuffer(data, dtype=np.int16)
                        samples = arr.reshape(1, -1) if arr.ndim == 1 else arr[:1, :]
                    except asyncio.TimeoutError:
                        sr = int(self._sample_rate) or 48000
                        if sr not in self._silence_cache:
                            num_samples = int(0.02 * sr)
                            self._silence_cache[sr] = np.zeros((1, num_samples), dtype=np.int16)
                        samples = self._silence_cache[sr]

                    frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
                    frame.sample_rate = int(self._sample_rate)
                    frame.pts = self._ts
                    frame.time_base = Fraction(1, int(self._sample_rate))
                    self._ts += samples.shape[1]
                    return frame

            self.pc.addTrack(MicAudioTrack(self._mic_queue, 48000))

            # Add adaptive video track
            if self.enable_video_input:
                self._video_track = AdaptiveVideoTrack(
                    width=self._target_video_width,
                    height=self._target_video_height,
                    fps=self._video_fps,
                    outer_conn=self
                )
                try:
                    self._video_sender = self.pc.addTrack(self._video_track)
                    logger.info(f"Video track added: {self._target_video_width}x{self._target_video_height}")
                except Exception as e:
                    logger.warning(f"Failed to add video track: {e}")
                    self._video_track = None

            # SDP exchange
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            answer_sdp = await self._exchange_sdp(offer.sdp)
            if not answer_sdp:
                raise RuntimeError("Failed to exchange SDP")

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

            # Wait for session creation
            try:
                await asyncio.wait_for(self.session_created_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Session creation timeout")

            await self._update_session()
            self._running = True
            logger.info("OpenAI Realtime session started")

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    async def _stop_session(self):
        """Stop the WebRTC session."""
        self._running = False
        try:
            if self._video_track:
                with contextlib.suppress(Exception):
                    self._video_track.stop()
            if self.dc:
                self.dc.close()
            await self.pc.close()
            if self._http:
                await self._http.aclose()
            logger.debug("Session stopped")
        except Exception as e:
            logger.error(f"Error stopping session: {e}")

    # Additional required methods (simplified versions)
    async def _get_session_token(self) -> Optional[str]:
        """Get session token from OpenAI."""
        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        session_config = {"model": self.model, "voice": self.voice}

        try:
            response = await self._http.post(url, headers=headers, json=session_config, timeout=15)
            response.raise_for_status()
            data = response.json()
            client_secret = data.get("client_secret")
            return client_secret.get("value") if isinstance(client_secret, dict) else client_secret
        except Exception as e:
            logger.error(f"Failed to get session token: {e}")
            return None

    async def _exchange_sdp(self, sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI."""
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/sdp"}
        url = f"https://api.openai.com/v1/realtime?model={self.model}"

        try:
            response = await self._http.post(url, headers=headers, content=sdp, timeout=20)
            response.raise_for_status()
            return response.text if response.text else None
        except Exception as e:
            logger.error(f"SDP exchange failed: {e}")
            return None

    async def _update_session(self):
        """Update session configuration."""
        event = {
            "type": "session.update",
            "session": {
                "instructions": self.system_instructions or "You are a helpful assistant.",
                "voice": self.voice,
                "turn_detection": {"type": "semantic_vad"} if self.turn_detection else None,
            },
        }
        await self._send_event(event)

    async def _handle_event(self, event: dict):
        """Handle OpenAI events."""
        event_type = event.get("type")

        if event_type == "session.created":
            self.openai_session_id = event["session"]["id"]
            self.session_created_event.set()
        elif event_type == "error":
            logger.error(f"OpenAI error: {event.get('error', {})}")

        # Notify callbacks
        for callback in self._event_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def _process_audio_track(self, track):
        """Process incoming audio from OpenAI."""
        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                    samples = frame.to_ndarray()
                    if samples.ndim == 2 and samples.shape[0] > 1:
                        samples = samples.mean(axis=0)
                    if samples.dtype != np.int16:
                        samples = (samples * 32767).astype(np.int16)

                    # Resample to 48kHz if needed
                    in_rate = getattr(frame, "sample_rate", 48000) or 48000
                    if in_rate != 48000:
                        samples = resample_audio(samples, in_rate, 48000).astype(np.int16)

                    audio_bytes = samples.tobytes()
                    for callback in self._audio_callbacks:
                        await callback(audio_bytes)

                except asyncio.TimeoutError:
                    continue
                except MediaStreamError:
                    logger.info("Audio stream ended")
                    break
        except Exception as e:
            logger.error(f"Error processing audio: {e}")


class Realtime(realtime.Realtime):
    """OpenAI Realtime API with improved video reliability."""

    def __init__(
            self,
            model: str = "gpt-realtime",
            api_key: Optional[str] = None,
            voice: str = "alloy",
            turn_detection: bool = True,
            instructions: Optional[str] = None,
            client: Optional[Any] = None,
            *,
            barge_in: bool = True,
            activity_threshold: int = 4000,
            silence_timeout_ms: int = 1200,
            enable_video_input: bool = False,
            video_fps: int = 1,
            video_width: int = 1280,
            video_height: int = 720,
            video_debug_enabled: bool = False,
    ):
        logger.info("Initializing improved Realtime with adaptive video track")
        logger.debug(f"Video settings: enabled={enable_video_input} fps={video_fps} size={video_width}x{video_height}")
        super().__init__(
            provider_name="openai-realtime-improved",
            model=model,
            instructions=instructions,
            voice=voice,
            provider_config={"turn_detection": turn_detection},
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.turn_detection = turn_detection
        self.system_prompt = instructions
        self.realtime = True
        self._connection: Optional[ImprovedRealtimeConnection] = None
        self._connection_lock = asyncio.Lock()
        self._playback_enabled = True

        # Video settings
        self.enable_video_input = enable_video_input
        self.video_fps = max(1, int(video_fps))
        self.video_width = max(1, int(video_width))
        self.video_height = max(1, int(video_height))
        self.video_debug_enabled = bool(video_debug_enabled)

        # Initialize required attributes
        self._is_connected = False
        self._ready_event = asyncio.Event()

        # Initialize output track attribute that Agent will set
        self._output_track = None
        logger.debug("Audio output track initialized")

        # Auto-start connection
        self._start_auto_connection()

    @property
    def output_track(self):
        return self._output_track

    @output_track.setter
    def output_track(self, track):
        self._output_track = track
        if track:
            logger.debug(f"Output track set: {track.__class__.__name__}")
        else:
            logger.debug("Output track cleared")

    def _start_auto_connection(self):
        """Auto-start connection helper."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_connection())
        except RuntimeError:
            pass

    async def _ensure_connection(self):
        """Ensure we have an active connection."""
        async with self._connection_lock:
            if not self._connection or not getattr(self, '_is_connected', False):
                await self._create_connection()

    async def _create_connection(self):
        """Create the WebRTC connection."""
        try:
            self._connection = ImprovedRealtimeConnection(
                api_key=self.api_key,
                model=self.model,
                voice=self.voice,
                turn_detection=self.turn_detection,
                system_instructions=self.system_prompt or self.instructions,
                enable_video_input=self.enable_video_input,
                video_fps=self.video_fps,
                target_video_width=self.video_width,
                target_video_height=self.video_height,
                video_debug_enabled=self.video_debug_enabled,
            )

            # Register handlers
            self._connection.on_event(self._handle_openai_event)
            self._connection.on_audio(self._handle_audio_output)

            # Start connection
            await self._connection._start_session()

            # Mark connected and emit events
            self._is_connected = True
            self._ready_event.set()

            event = RealtimeConnectedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                provider="openai-improved",
                session_config={
                    "model": self.model,
                    "voice": self.voice,
                    "turn_detection": self.turn_detection,
                },
                capabilities=["text", "audio"],
            )
            register_global_event(event)
            self.emit("connected", event)

        except Exception as e:
            self._emit_error_event(e, "connection")
            raise

    async def _handle_openai_event(self, event: dict):
        """Handle OpenAI events."""
        event_type = event.get("type")

        if event_type == "response.audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                self._emit_transcript_event(
                    text=transcript,
                    user_metadata={"role": "assistant", "source": "openai-improved"},
                )

        elif event_type == "error":
            error = event.get("error", {})
            self._emit_error_event(
                error=Exception(error.get("message", "Unknown error")),
                context=f"openai_improved_event: {error.get('code', 'unknown')}",
            )

    async def _handle_audio_output(self, audio_bytes: bytes):
        """Handle audio output from OpenAI."""
        if not audio_bytes:
            return

        logger.debug(f"Received {len(audio_bytes)} bytes from OpenAI")

        # Emit audio output event for listeners
        listeners_fn = getattr(self, "listeners", None)
        has_listeners = bool(listeners_fn("audio_output")) if callable(listeners_fn) else False
        if has_listeners:
            self._emit_audio_output_event(audio_data=audio_bytes, sample_rate=48000)
            logger.debug("Emitted audio_output event")

        # Write to output track for remote participants to hear
        output_track = getattr(self, "output_track", None)
        if output_track is not None and self._playback_enabled:
            try:
                await output_track.write(audio_bytes)
                logger.debug(f"Wrote {len(audio_bytes)} bytes to output track")
            except Exception as e:
                logger.error(f"Failed to write to output track: {e}")
        else:
            if output_track is None:
                logger.warning("No output_track set - audio will not be heard")
            if not self._playback_enabled:
                logger.debug("Audio playback is disabled")

    @asynccontextmanager
    async def connect(self, call: Call, agent_user_id: str):
        """Connection context manager."""
        logger.info(f"Connecting OpenAI Realtime for call {call.id}")
        await self._ensure_connection()
        try:
            yield self._connection
        finally:
            pass

    async def send_audio_pcm(self, pcm_data, target_rate: int = 48000):
        """Send PCM audio to OpenAI."""
        await self._ensure_connection()
        if not self._connection:
            return

        # Extract audio bytes
        audio_bytes = None
        samples = getattr(pcm_data, "samples", None)
        if samples is not None:
            if not isinstance(samples, (bytes, bytearray)):
                arr = np.asarray(samples)
                if arr.size == 0:
                    return
                if arr.dtype != np.int16:
                    arr = arr.astype(np.int16)
                audio_bytes = arr.tobytes()
            else:
                audio_bytes = bytes(samples)
        elif isinstance(pcm_data, bytes):
            audio_bytes = pcm_data

        if audio_bytes:
            src_rate = int(getattr(pcm_data, "sample_rate", 48000))
            self._emit_audio_input_event(audio_data=audio_bytes, sample_rate=src_rate)

            # Send in chunks
            frame_bytes = int(0.02 * src_rate) * 2  # 20ms
            for i in range(0, len(audio_bytes), frame_bytes):
                chunk = audio_bytes[i : i + frame_bytes]
                if chunk:
                    if len(chunk) < frame_bytes:
                        chunk += b"\\x00" * (frame_bytes - len(chunk))
                    await self._connection.send_audio(chunk, sample_rate=src_rate)

    async def send_text(self, text: str):
        """Send text message."""
        await self._ensure_connection()
        self._emit_transcript_event(text=text, user_metadata={"role": "user"})
        if not self._connection:
            raise RuntimeError("No connection")
        await self._connection.send_text(text)

    async def start_video_sender(self, track: MediaStreamTrack, fps: Optional[int] = None):
        """Start video forwarding."""
        logger.info(f"Starting video sender with track: {track.__class__.__name__}")
        await self._ensure_connection()
        if self._connection:
            await self._connection.start_video_sender(track, fps or self.video_fps)
        else:
            logger.error("No connection available for video forwarding")

    async def stop_video_sender(self):
        """Stop video forwarding."""
        if self._connection:
            await self._connection.stop_video_sender()

    async def _close_impl(self):
        """Close the connection."""
        if self._connection:
            try:
                await self._connection._stop_session()
            except Exception as e:
                logger.error(f"Error closing: {e}")
            finally:
                self._connection = None

    async def simple_response(self, *, text: str, timeout: Optional[float] = 30.0):
        """Simple response implementation."""
        return await super().simple_response(text=text, timeout=timeout)

    async def create_response(self, *args, **kwargs):
        """Compatibility wrapper."""
        text = kwargs.get("input", args[0] if args else "")
        rt_resp = await self.simple_response(text=text)
        return LLMResponse(original=rt_resp.original, text=rt_resp.text)
