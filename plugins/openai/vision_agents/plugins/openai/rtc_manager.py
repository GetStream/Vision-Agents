import asyncio
import json
import time
from typing import Any, Optional, Callable, cast
from os import getenv

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from httpx import AsyncClient, HTTPStatusError
import logging
from getstream.video.rtc.track_util import PcmData

from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamTrack
from fractions import Fraction
import numpy as np
from av import AudioFrame, VideoFrame

from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger(__name__)

# OpenAI Realtime endpoints
OPENAI_REALTIME_BASE = "https://api.openai.com/v1/realtime"
OPENAI_SESSIONS_URL = f"{OPENAI_REALTIME_BASE}/sessions"


class RealtimeAudioTrack(AudioStreamTrack):
    """Minimal audio track without a queue.

    - Generates 20 ms mono PCM16 silence by default
    - Accepts optional push-based PCM via set_input(bytes, sample_rate)
    - Drops old input if new input arrives (no buffering)

    TODO: why do we need this instead of forwarding from 1 AudioStreamTrack to another?
    """

    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._sample_rate = int(sample_rate)
        self._ts = 0
        self._latest_chunk: Optional[bytes] = None
        self._silence_cache: dict[int, np.ndarray] = {}

    def set_input (self, pcm_data: bytes, sample_rate: Optional[int] = None) -> None:
        if not pcm_data:
            return
        if sample_rate is not None:
            self._sample_rate = int(sample_rate)
        self._latest_chunk = bytes(pcm_data)

    async def recv(self):
        # Pace roughly at 20 ms per frame
        await asyncio.sleep(0.02)

        sr = int(self._sample_rate) if self._sample_rate else 48000
        samples_per_frame = int(0.02 * sr)

        chunk = self._latest_chunk
        if chunk:
            # Consume and clear the latest pushed chunk
            self._latest_chunk = None
            arr = np.frombuffer(chunk, dtype=np.int16)
            if arr.ndim == 1:
                samples = arr.reshape(1, -1)
            else:
                samples = arr[:1, :]  # type: ignore[assignment]
            # Pad or truncate to exactly one 20 ms frame
            # TODO: why do we have this? this is handled by the audio track queue
            needed = samples_per_frame
            have = samples.shape[1]
            if have < needed:
                pad = np.zeros((1, needed - have), dtype=np.int16)
                samples = np.concatenate([samples, pad], axis=1)  # type: ignore[assignment]
            elif have > needed:
                samples = samples[:, :needed]  # type: ignore[assignment]
        else:
            cached = self._silence_cache.get(sr)
            if cached is None:
                cached = np.zeros((1, samples_per_frame), dtype=np.int16)
                self._silence_cache[sr] = cached
            samples = cached

        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = sr
        frame.pts = self._ts
        frame.time_base = Fraction(1, sr)
        self._ts += samples.shape[1]
        return frame


class StreamVideoForwardingTrack(VideoStreamTrack):
    """Track that forwards frames from Stream Video to OpenAI.
    TODO: why do we have this forwarding track, when there is the video_forwarder
    """
    
    kind = "video"
    
    def __init__(self, source_track: MediaStreamTrack, fps: int = 1, shared_forwarder=None):
        super().__init__()
        self._source_track = source_track
        self._fps = max(1, fps)
        self._interval = 1.0 / self._fps
        self._ts = 0
        self._last_frame_time = 0.
        self._frame_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._is_active = True
        self._last_successful_frame_time = time.monotonic()
        self._health_check_interval = 10.0  # Check health every 10 seconds
        self._last_health_check = time.monotonic()
        self._forwarder: Optional[VideoForwarder] = None
        self._shared_forwarder = shared_forwarder
        self._started: bool = False

        # Rate limiting for inactive track warnings
        self._last_inactive_warning = 0.
        self._inactive_warning_interval = 30.0  # Only warn every 30 seconds
        
        if shared_forwarder:
            logger.info(f"🎥 StreamVideoForwardingTrack initialized with SHARED forwarder: fps={fps}, interval={self._interval:.3f}s")
        else:
            logger.info(f"🎥 StreamVideoForwardingTrack initialized: fps={fps}, interval={self._interval:.3f}s (frame limiting DISABLED for performance)")
    
    async def start(self) -> None:
        if self._started:
            return
        
        if self._shared_forwarder is not None:
            # Use the shared forwarder
            self._forwarder = self._shared_forwarder
            logger.info(f"🎥 OpenAI using shared VideoForwarder at {self._fps} FPS")
        else:
            # Create our own VideoForwarder with the input source track (legacy behavior)
            self._forwarder = VideoForwarder(self._source_track, max_buffer=5, fps=self._fps)  # type: ignore[arg-type]
            await self._forwarder.start()
        
        self._started = True

    async def recv(self):
        """Pull latest frame from forwarder's latest-N buffer and return it to WebRTC."""
        if not self._started:
            await self.start()

        if not self._is_active:
            # Rate limit warnings to avoid spam
            now = time.monotonic()
            if now - self._last_inactive_warning > self._inactive_warning_interval:
                logger.warning("🎥 StreamVideoForwardingTrack is no longer active, returning black frame")
                self._last_inactive_warning = now
            return self._generate_black_frame()

        now = time.monotonic()

        # Health check: detect if track has been dead for too long
        if now - self._last_health_check > self._health_check_interval:
            self._last_health_check = now
            if now - self._last_successful_frame_time > 30.0:  # No frames for 30 seconds
                logger.error("🎥 StreamVideoForwardingTrack health check failed - no frames for 30+ seconds")
                self._is_active = False
                return self._generate_black_frame()

        try:
            # Pull the newest frame, coalescing backlog
            timeout = max(0.01, self._interval * 2)
            assert self._forwarder is not None
            frame = await self._forwarder.next_frame(timeout=timeout)

            # Reset error counters on success
            self._consecutive_errors = 0
            self._frame_count += 1
            self._last_successful_frame_time = now

            # Convert to rgb24 if needed
            if frame.format.name != "rgb24":
                try:
                    frame = frame.reformat(format="rgb24")
                except Exception as e:
                    logger.warning(f"🎥 Frame format conversion failed: {e}, using original")

            # Update timing for WebRTC
            frame.pts = self._ts
            frame.time_base = Fraction(1, self._fps)
            self._ts += 1
            self._last_frame_time = time.monotonic()

            return frame

        except asyncio.TimeoutError:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.error(f"🎥 StreamVideoForwardingTrack circuit breaker triggered - {self._consecutive_errors} consecutive timeouts")
                self._is_active = False
            return self._generate_black_frame()
        except Exception as e:
            self._consecutive_errors += 1
            self._error_count += 1
            logger.error(f"❌ FRAME ERROR: frame_id={self._frame_count} (error #{self._error_count}, consecutive={self._consecutive_errors}): {e}")
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.error(f"🎥 StreamVideoForwardingTrack circuit breaker triggered - {self._consecutive_errors} consecutive errors")
                self._is_active = False
            return self._generate_black_frame()
    
    def _generate_black_frame(self) -> VideoFrame:
        """Generate a black frame as fallback."""
        black_array = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(black_array, format="rgb24")
        frame.pts = self._ts
        frame.time_base = Fraction(1, self._fps)
        self._ts += 1
        return frame
    
    def stop(self):
        """Stop the forwarding track and forwarder."""
        logger.info(f"🎥 StreamVideoForwardingTrack stopped after {self._frame_count} frames, {self._error_count} errors")
        try:
            if self._forwarder is not None:
                asyncio.create_task(self._forwarder.stop())
                self._forwarder = None
        except Exception:
            pass
        super().stop()


class RTCManager:
    """Manages WebRTC connection to OpenAI's Realtime API.

    Handles the low-level WebRTC peer connection, audio/video streaming,
    and data channel communication with OpenAI's servers.
    """

    def __init__(self, model: str, voice: str, send_video: bool):
        """Initialize the RTC manager.

        Args:
            model: OpenAI model to use for the session (e.g., "gpt-realtime").
            voice: Voice to use for audio responses (e.g., "marin", "alloy").
            send_video: Whether to enable video track negotiation for potential video input.
        """
        self.api_key = getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.token = ""
        self.session_info: Optional[dict] = None  # Store session information
        self.pc = RTCPeerConnection()
        self.data_channel: Optional[RTCDataChannel] = None
        self._mic_track: RealtimeAudioTrack = RealtimeAudioTrack(48000)
        self._audio_callback: Optional[Callable[[bytes], Any]] = None
        self._event_callback: Optional[Callable[[dict], Any]] = None
        self._data_channel_open_event: asyncio.Event = asyncio.Event()
        self.send_video = send_video
        self._video_track: Optional[VideoStreamTrack] = None
        self._video_sender_task: Optional[asyncio.Task] = None
        self._forwarding_track: Optional[StreamVideoForwardingTrack] = None
        self.instructions: Optional[str] = None

    async def connect(self) -> None:
        """Establish WebRTC connection to OpenAI's Realtime API.

        Sets up the peer connection, negotiates audio and video tracks,
        and establishes the data channel for real-time communication.
        """
        self.token = await self._get_session_token()
        await self._add_data_channel()

        await self._set_audio_track()

        if self.send_video:
            await self._set_video_track()

        @self.pc.on("track")
        async def on_track(track):
            await self._handle_added_track(track)

        answer_sdp = await self._setup_sdp_exchange()

        # Set the remote SDP we got from OpenAI
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)
        logger.info("Remote description set; WebRTC established")


    async def _get_session_token(self) -> str:
        url = OPENAI_SESSIONS_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # TODO: replace with regular openai client SDK when support for this endpoint is added
        # TODO: voice is not the right param or typing is wrong here
        # Not quite: RealtimeSessionCreateRequestParam
        payload = {"model": self.model, "voice": self.voice}  # type: ignore[typeddict-unknown-key]
        if self.instructions:
            payload["instructions"] = self.instructions

        async with AsyncClient() as client:
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, json=payload, timeout=15)
                    resp.raise_for_status()
                    data: dict = resp.json()
                    secret = data.get("client_secret", {})
                    return secret.get("value")
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"Failed to get OpenAI Realtime session token: {e}")
                    raise
            raise Exception("Failed to get OpenAI Realtime session token")

    async def _add_data_channel(self) -> None:
        # Add data channel
        self.data_channel = self.pc.createDataChannel("oai-events")

        @self.data_channel.on("open")
        async def on_open():
            self._data_channel_open_event.set()

            # Immediately switch to semantic VAD so it's active before the user speaks
            await self._send_event(
                    {
                        "type": "session.update",
                        "session": {
                            "turn_detection": {
                                "type": "semantic_vad"
                            }
                        },
                    }
                )
            # Session information will be automatically stored when session.created event is received
            logger.info("Requested semantic_vad via session.update")

        @self.data_channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                asyncio.create_task(self._handle_event(data))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")

    async def _set_audio_track(self) -> None:
        self.pc.addTrack(self._mic_track)

    async def _set_video_track(self) -> None:
        class RealtimeVideoTrack(VideoStreamTrack):
            kind = "video"

            def __init__(self):
                super().__init__()
                self._ts = 0

            async def recv(self):
                await asyncio.sleep(0.02)
                width = 640
                height = 480
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                rgb[:, :, 2] = 255  # Blue in RGB
                frame = VideoFrame.from_ndarray(rgb, format="rgb24", channel_last=True)
                frame.pts = self._ts
                frame.time_base = Fraction(1, 50)
                self._ts += 1
                return frame

        self._video_track = RealtimeVideoTrack()
        self._video_sender = self.pc.addTrack(self._video_track)
        # Keep a handle to the currently active source (if any) for diagnostics / control
        self._active_video_source: Optional[MediaStreamTrack] = None


    async def send_audio_pcm(self, pcm_data: PcmData) -> None:
        """Send raw PCM audio data to OpenAI.

        Args:
            pcm_data: PCM audio data containing samples and sample rate.
        """
        if not self._mic_track:
            return
        try:
            sr = pcm_data.sample_rate or 48000
            arr = pcm_data.samples
            if arr.size == 0:
                return
            if arr.ndim == 2 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            if arr.dtype != np.int16:
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
            self._mic_track.set_input(arr.tobytes(), sr)
        except Exception as e:
            logger.error(f"Failed to push mic audio: {e}")


    async def send_text(self, text: str, role: str = "user"):
        """Send a text message to OpenAI.
        
        Args:
            text: The text message to send.
            role: Message role. Defaults to "user".
        """
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": text}],
            },
        }
        await self._send_event(event)
        # Explicitly request audio response for this turn using top-level fields
        await self._send_event(
            {
                "type": "response.create",
            }
        )

    async def _send_event(self, event: dict):
        """Send an event through the data channel."""
        if not self.data_channel:
            logger.warning("Data channel not ready, cannot send event")
            return

        try:
            # Ensure the data channel is open before sending
            if not self._data_channel_open_event.is_set():
                try:
                    await asyncio.wait_for(self._data_channel_open_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Data channel not open after timeout; dropping event")
                    return

            if self.data_channel.readyState and self.data_channel.readyState != "open":
                logger.warning(f"Data channel state is '{self.data_channel.readyState}', cannot send event")

            message_json = json.dumps(event)
            self.data_channel.send(message_json)
            logger.debug(f"Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def start_video_sender(self, stream_video_track: MediaStreamTrack, fps: int = 1, shared_forwarder=None) -> None:
        """Replace dummy video track with the actual Stream Video forwarding track.

        This creates a forwarding track that reads frames from the Stream Video track
        and forwards them through the OpenAI WebRTC connection.

        Args:
            stream_video_track: Video track to forward to OpenAI.
            fps: Target frames per second.
            shared_forwarder: Optional shared VideoForwarder to use instead of creating a new one.
        """
        
        try:
            if not self.send_video:
                logger.error("❌ Video sending not enabled for this session")
                raise RuntimeError("Video sending not enabled for this session")
            if self._video_sender is None:
                logger.error("❌ Video sender not available; was video track negotiated?")
                raise RuntimeError("Video sender not available; was video track negotiated?")
            
            # Stop any existing video sender task
            if self._video_sender_task is not None:
                logger.info("🎥 Stopping existing video sender task...")
                self._video_sender_task.cancel()
                try:
                    await self._video_sender_task
                except asyncio.CancelledError:
                    pass
                logger.info("🎥 Existing video sender task stopped")
            
            # Create forwarding track and start its forwarder
            forwarding_track = StreamVideoForwardingTrack(stream_video_track, fps, shared_forwarder=shared_forwarder)
            await forwarding_track.start()
            
            # Replace the dummy track with the forwarding track
            try:
                logger.info("🎥 Replacing OpenAI dummy track with StreamVideoForwardingTrack")
                self._video_sender.replaceTrack(forwarding_track)
                self._forwarding_track = forwarding_track
                self._active_video_source = stream_video_track
                logger.info(f"✅ Successfully replaced OpenAI track with Stream Video forwarding (fps={fps})")
            except Exception as replace_error:
                logger.error(f"❌ Failed to replace video track: {replace_error}")
                logger.error(f"❌ Replace error type: {type(replace_error).__name__}")
                raise RuntimeError(f"Track replacement failed: {replace_error}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start video sender: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            raise

    async def stop_video_sender(self) -> None:
        """Stop video forwarding and restore the dummy negotiated video track (blue/black frames)."""
        try:
            # Stop the video forwarding task
            if self._video_sender_task is not None:
                self._video_sender_task.cancel()
                try:
                    await self._video_sender_task
                except asyncio.CancelledError:
                    pass
                logger.info("✅ Video forwarding task stopped")
            # Stop forwarding track/forwarder if present
            if self._forwarding_track is not None:
                try:
                    self._forwarding_track.stop()
                except Exception:
                    pass
                self._forwarding_track = None
            
            if self._video_sender is None:
                logger.warning("No video sender available to stop")
                return
            
            # Replace track with proper error handling
            try:
                if self._video_track is None:
                    # If we have no base track, detach the current track
                    self._video_sender.replaceTrack(None)
                    logger.info("✅ Video sender detached (no base track)")
                else:
                    self._video_sender.replaceTrack(self._video_track)
                    logger.info(f"✅ Video sender reverted to dummy track: {type(self._video_track).__name__}")
                
                self._active_video_source = None
            except Exception as replace_error:
                logger.error(f"❌ Failed to revert video track: {replace_error}")
                raise RuntimeError(f"Track reversion failed: {replace_error}")
                
        except Exception as e:
            logger.error(f"❌ Failed to stop video sender: {e}")
            raise

    async def _setup_sdp_exchange(self) -> str:
        # Create local offer and exchange SDP
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        answer_sdp = await self._exchange_sdp(offer.sdp)
        if not answer_sdp:
            raise RuntimeError("Failed to get remote SDP from OpenAI")
        return answer_sdp

    async def _exchange_sdp(self, local_sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI."""
        # IMPORTANT: Use the ephemeral client secret token from session.create
        token = self.token or self.api_key
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/sdp",
            "OpenAI-Beta": "realtime=v1",
        }
        url = f"{OPENAI_REALTIME_BASE}?model={self.model}"

        try:
            async with AsyncClient() as client:
                response = await client.post(url, headers=headers, content=local_sdp, timeout=20)
                response.raise_for_status()
                return response.text if response.text else None
        except HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            logger.error(f"SDP exchange failed: {e}; body={body}")
            raise
        except Exception as e:
            logger.error(f"SDP exchange failed: {e}")
            raise

    # When you get a remote track (OpenAI) we write the audio from the track on the call.
    async def _handle_added_track(self, track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            logger.info("Remote audio track attached; starting audio reader")

            async def _reader():
                while True:
                    try:
                        frame: AudioFrame = cast(AudioFrame, await asyncio.wait_for(track.recv(), timeout=1.0))
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.debug(f"Remote audio track ended or error: {e}")
                        break

                    try:
                        # TODO: why not use the utility methods for this on PcmData?
                        # TODO: why do we even need this, audio tracks automatically handle it in some cases
                        samples = frame.to_ndarray()
                        if samples.ndim == 2 and samples.shape[0] > 1:
                            samples = samples.mean(axis=0)
                        if samples.dtype != np.int16:
                            samples = (samples * 32767).astype(np.int16)
                        audio_bytes = samples.tobytes()
                        cb = self._audio_callback
                        if cb is not None:
                            await cb(audio_bytes)
                    except Exception as e:
                        logger.debug(f"Failed to process remote audio frame: {e}")
            asyncio.create_task(_reader())
            

    async def _handle_event(self, event: dict) -> None:
        """Minimal event handler for data channel messages."""
        cb = self._event_callback
        if cb is not None:
            try:
                await cb(event)
            except Exception as e:
                logger.debug(f"Event callback error: {e}")

        # Store session information when we receive session.created event
        # FIXME Typing 
        if event.get("type") == "session.created" and "session" in event:
            self.session_info = event["session"]
            logger.debug(f"Stored session info: {self.session_info}")

    async def request_session_info(self) -> None:
        """Request and log current session information.

        Note:
            Session information is automatically stored when session.created event is received.
            This method only logs the stored information.
        """
        if self.session_info:
            logger.info(f"Current session info: {self.session_info}")
        else:
            logger.info("No session information available yet. Waiting for session.created event.")

    def set_audio_callback(self, callback: Callable[[bytes], Any]) -> None:
        """Set callback for receiving audio data from OpenAI.

        Args:
            callback: Function that receives raw audio bytes from OpenAI responses.
        """
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable[[dict], Any]) -> None:
        """Set callback for receiving events from OpenAI.

        Args:
            callback: Function that receives event dicts from the OpenAI data channel.
        """
        self._event_callback = callback

    async def _forward_video_frames(self, source_track: MediaStreamTrack, fps: int) -> None:
        """Forward video frames from user's track to OpenAI via WebRTC.
        
        This method reads frames from the user's video track and forwards them
        through the WebRTC connection to OpenAI for processing.
        """
        interval = max(0.01, 1.0 / max(1, fps))
        frame_count = 0
        
        try:
            logger.info(f"🎥 Starting video frame forwarding loop (fps={fps}, interval={interval:.3f}s)")
            logger.info(f"🎥 Source track: {type(source_track).__name__}, kind={getattr(source_track, 'kind', 'unknown')}")
            
            while True:
                try:
                    # Read frame from user's video track
                    logger.debug(f"🎥 Attempting to read frame #{frame_count + 1} from user track...")
                    frame: VideoFrame = cast(VideoFrame, await asyncio.wait_for(source_track.recv(), timeout=1.0))
                    frame_count += 1
                    
                    # Log frame details
                    logger.info(f"🎥 SUCCESS: Read frame #{frame_count} from user track!")
                    logger.info(f"🎥 Frame details: {frame.width}x{frame.height}, format={frame.format}, pts={frame.pts}")
                    
                    # The frame is automatically forwarded through the WebRTC connection
                    # since we replaced the track with replaceTrack()
                    logger.debug(f"🎥 Frame #{frame_count} automatically forwarded via WebRTC")
                    
                    # Throttle frame rate
                    await asyncio.sleep(interval)
                    
                except asyncio.TimeoutError:
                    logger.warning(f"🎥 Timeout waiting for frame #{frame_count + 1} from user track")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error reading video frame #{frame_count + 1}: {e}")
                    logger.error(f"❌ Exception type: {type(e).__name__}")
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"🎥 Video forwarding task cancelled after {frame_count} frames")
        except Exception as e:
            logger.error(f"❌ Video forwarding task failed after {frame_count} frames: {e}")
        finally:
            logger.info(f"🎥 Video forwarding task ended. Total frames processed: {frame_count}")

    async def close(self) -> None:
        """Close the WebRTC connection and clean up resources."""
        try:
            # Clean up video sender task
            if self._video_sender_task is not None:
                self._video_sender_task.cancel()
                try:
                    await self._video_sender_task
                except asyncio.CancelledError:
                    pass
            
            if self.data_channel is not None:
                try:
                    self.data_channel.close()
                except Exception:
                    pass
                self.data_channel = None
            if self._mic_track is not None:
                try:
                    self._mic_track.stop()
                except Exception:
                    pass
            await self.pc.close()
        except Exception as e:
            logger.debug(f"RTCManager close error: {e}")
