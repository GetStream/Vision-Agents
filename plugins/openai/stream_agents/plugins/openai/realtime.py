"""OpenAI Realtime API integration with WebRTC."""

import asyncio
from contextlib import asynccontextmanager
import contextlib
from fractions import Fraction
import json
import hashlib
import time
import os
import logging
import os
import httpx
import traceback
from typing import Any, AsyncIterator, Optional
 

from aiortc import (
    AudioStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaStreamError, MediaStreamTrack
from av import AudioFrame, VideoFrame
from getstream.audio.utils import resample_audio
from getstream.video.call import Call
import numpy as np
from stream_agents.core.events import RealtimeConnectedEvent, register_global_event
from stream_agents.core.llm import realtime
from stream_agents.core.llm.llm import LLMResponse

 
logger = logging.getLogger(__name__)


class RealtimeConnection:
    """Internal WebRTC connection handler for OpenAI Realtime API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-realtime",
        voice: str = "alloy",
        turn_detection: bool = True,
        system_instructions: Optional[str] = None,
        enable_video_input: bool = False,
        video_fps: int = 1,
        *,
        target_video_width: int = 1280,
        target_video_height: int = 720,
        video_debug_enabled: bool = False,
        save_snapshots_enabled: bool = False,
        snapshot_interval_sec: float = 3.0,
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
        # Queue and media track for microphone audio over WebRTC
        self._mic_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        # Backpressure params: keep queue tiny to avoid latency buildup
        self._max_queue_frames: int = 3  # ~60ms at 20ms per frame
        self._frames_dropped: int = 0
        self._mic_track: Optional[AudioStreamTrack] = None
        # Remote audio receive counters (removed verbose logging)
        # Shared HTTP client for REST calls to OpenAI
        self._http: Optional[httpx.AsyncClient] = None
        # Video negotiation/sender state
        self._black_video_track: Optional[MediaStreamTrack] = None
        self._forward_video_track: Optional[MediaStreamTrack] = None
        self._video_sender = None
        self._last_frame_rgb: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0
        self._out_video_prev: dict = {}
        self._snapshot_task: Optional[asyncio.Task] = None
        self._snapshot_dir: str = os.path.join(os.getcwd(), "snapshots", "openai_forwarded")
        self._snapshot_interval_sec: float = max(0.5, float(snapshot_interval_sec))
        # Debug / diagnostics feature flags (configurable by client)
        self._video_debug_enabled: bool = bool(video_debug_enabled)
        self._save_snapshots_enabled: bool = bool(save_snapshots_enabled)
        # Target dimensions for optional downscale before sending
        self._target_video_width: int = max(1, int(target_video_width))
        self._target_video_height: int = max(1, int(target_video_height))
        # Cache of last seen connection state
        self._last_conn_state: Optional[str] = None
        # Video diagnostics
        self._video_stall_count: int = 0
        self._saved_first_snapshot: bool = False
        # Provider session id (from OpenAI session.created)
        self.openai_session_id: Optional[str] = None
        # Cache active video sender
        self._active_video_sender = None

    async def __aenter__(self):
        """Start the WebRTC session with OpenAI."""
        await self._start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the WebRTC session."""
        await self._stop_session()

    def on_audio(self, callback):
        """Register a callback for audio data from OpenAI."""
        self._audio_callbacks.append(callback)

    def on_event(self, callback):
        """Register a callback for events from OpenAI."""
        self._event_callbacks.append(callback)

    async def send_audio(self, audio_data: bytes, sample_rate: int = 48000):
        """Enqueue audio bytes to the microphone media track (PCM16 mono).

        The sample_rate is carried with the frame to avoid Python-side resampling.
        """
        if not audio_data:
            return
        # Aggressive backpressure: keep only a few most recent frames
        while self._mic_queue.qsize() >= self._max_queue_frames:
            try:
                _ = self._mic_queue.get_nowait()
                self._frames_dropped += 1
            except asyncio.QueueEmpty:
                break
        # Store (bytes, sample_rate) tuple
        self._mic_queue.put_nowait((audio_data, sample_rate))
        if self._frames_dropped and (self._frames_dropped % 50 == 0):
            logger.debug(
                f"mic_queue: dropped_frames={self._frames_dropped} size={self._mic_queue.qsize()}"
            )

    async def send_text(self, text: str):
        """Send a text message to OpenAI."""
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
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
        if not self.dc:
            logger.warning("Data channel not ready, cannot send event")
            return

        try:
            message_json = json.dumps(event)
            self.dc.send(message_json)
            logger.debug(f"Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _get_session_token(self) -> Optional[str]:
        """Get a session token from OpenAI using async httpx."""
        url = "https://api.openai.com/v1/realtime/sessions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        session_config = {"model": self.model, "voice": self.voice}
        client = self._http
        if client is None:
            async with httpx.AsyncClient() as tmp_client:
                return await self._get_session_token_with_client(
                    tmp_client, url, headers, session_config
                )
        return await self._get_session_token_with_client(
            client, url, headers, session_config
        )

    async def _get_session_token_with_client(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        session_config: dict,
    ) -> Optional[str]:
        for attempt in range(3):
            try:
                response = await client.post(
                    url, headers=headers, json=session_config, timeout=15
                )
                response.raise_for_status()
                data = response.json()
                logger.info("Successfully obtained OpenAI session token")
                client_secret = data.get("client_secret")
                if isinstance(client_secret, dict):
                    return client_secret.get("value")
                return client_secret
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"Failed to get session token (attempt {attempt+1}/3): {e}"
                )
                status = e.response.status_code if e.response is not None else 0
                body = e.response.text if e.response is not None else ""
                logger.error(f"Response status: {status}")
                logger.error(f"Response body: {body}")
                if status and 500 <= status < 600 and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
                logger.error(
                    f"Failed to get session token (attempt {attempt+1}/3): {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None

    async def _exchange_sdp(self, sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI using async httpx."""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/sdp",
        }
        url = f"https://api.openai.com/v1/realtime?model={self.model}"
        client = self._http
        if client is None:
            async with httpx.AsyncClient() as tmp_client:
                return await self._exchange_sdp_with_client(tmp_client, url, headers, sdp)
        return await self._exchange_sdp_with_client(client, url, headers, sdp)

    async def _exchange_sdp_with_client(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: dict,
        sdp: str,
    ) -> Optional[str]:
        for attempt in range(3):
            try:
                response = await client.post(
                    url, headers=headers, content=sdp, timeout=20
                )
                response.raise_for_status()
                logger.info("SDP exchange successful")
                if response.text:
                    return response.text
                logger.error("Received empty SDP response")
                return None
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"SDP exchange failed (attempt {attempt+1}/3): {e}"
                )
                status = e.response.status_code if e.response is not None else 0
                body = e.response.text if e.response is not None else ""
                logger.error(f"Response status: {status}")
                logger.error(f"Response body: {body}")
                if status and 500 <= status < 600 and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPError) as e:
                logger.error(
                    f"SDP exchange failed (attempt {attempt+1}/3): {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return None

    async def _start_session(self):
        """Start the WebRTC session with OpenAI."""
        try:
            # Ensure shared HTTP client exists
            if self._http is None:
                self._http = httpx.AsyncClient()
            # Get session token
            self.token = await self._get_session_token()
            if not self.token:
                raise RuntimeError("Failed to obtain session token")

            # Set up peer connection event handlers
            @self.pc.on("track")
            async def on_track(track):
                if track.kind == "audio":
                    track_id = getattr(track, "id", "<no-id>")
                    logger.info(
                        f"on_track: remote audio track attached id={track_id}"
                    )
                    # Extra diagnostics for the audio track in debug mode
                    print(
                        "[OpenAI Realtime] Remote audio track attached:",
                        {
                            "id": getattr(track, "id", None),
                            "kind": getattr(track, "kind", None),
                            "class": track.__class__.__name__,
                            "readyState": getattr(track, "readyState", None),
                        },
                    )
                    asyncio.create_task(self._process_audio_track(track))
                else:
                    logger.info(f"on_track: non-audio track received kind={track.kind}")

            # Connection state diagnostics
            # Minimal connection state diagnostics
            @self.pc.on("connectionstatechange")
            async def on_conn_state_change():
                self._last_conn_state = self.pc.connectionState
                logger.info(f"pc.connectionState={self.pc.connectionState}")

            # Create data channel for events
            self.dc = self.pc.createDataChannel("oai-events")

            @self.dc.on("open")
            def on_open():
                logger.info("Data channel opened")
                # Fallback: if provider doesn't send session.created promptly, consider session ready
                if not self.session_created_event.is_set():
                    self.session_created_event.set()

            @self.dc.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    asyncio.create_task(self._handle_event(data))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")

            # Add a real microphone audio track to send audio to OpenAI over RTP
            from aiortc.mediastreams import AudioStreamTrack as _AudioStreamTrack

            class MicAudioTrack(_AudioStreamTrack):
                """Audio track that pulls 48 kHz PCM16 frames from an asyncio.Queue."""

                kind = "audio"

                def __init__(self, queue: asyncio.Queue, sample_rate: int = 48000):
                    super().__init__()
                    self._queue = queue
                    self._sample_rate = sample_rate  # last-seen or default
                    self._ts = 0
                    # Cache of 20ms mono int16 silence arrays per sample rate
                    self._silence_cache = {}

                async def recv(self):
                    # Try to get the next chunk quickly; emit 20 ms silence if none
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=0.02)
                        if isinstance(item, tuple):
                            data, sr = item
                        else:
                            data, sr = item, self._sample_rate
                        # Update current sample rate to what we received
                        self._sample_rate = int(sr) if sr else self._sample_rate
                        arr = np.frombuffer(data, dtype=np.int16)
                        if arr.ndim == 1:
                            samples = arr.reshape(1, -1)
                        else:
                            samples = arr[:1, :]
                    except asyncio.TimeoutError:
                        sr = int(self._sample_rate) if self._sample_rate else 48000
                        cached = self._silence_cache.get(sr)
                        if cached is None:
                            num_samples = int(0.02 * sr)
                            cached = np.zeros((1, num_samples), dtype=np.int16)
                            self._silence_cache[sr] = cached
                        samples = cached

                    frame = AudioFrame.from_ndarray(
                        samples, format="s16", layout="mono"
                    )
                    sr = int(self._sample_rate)
                    frame.sample_rate = sr
                    frame.pts = self._ts
                    frame.time_base = Fraction(1, sr)
                    self._ts += samples.shape[1]
                    return frame

            self._mic_track = MicAudioTrack(self._mic_queue, 48000)
            self.pc.addTrack(self._mic_track)

            # Optionally pre-negotiate a sendonly video m-line using a blank video track
            if self.enable_video_input:
                from aiortc.mediastreams import VideoStreamTrack as _VideoStreamTrack

                class BlackVideoTrack(_VideoStreamTrack):
                    kind = "video"

                    def __init__(self, width: int = 1280, height: int = 720, fps: int = 1):
                        super().__init__()
                        self._width = width
                        self._height = height
                        self._fps = max(1, int(fps))
                        self._last_ts = None

                    async def recv(self):
                        # Generate a black frame
                        import numpy as _np
                        import time as _time

                        # Pace output frames to the configured FPS
                        interval = 1.0 / float(self._fps)
                        now = _time.monotonic()
                        if self._last_ts is not None:
                            remaining = (self._last_ts + interval) - now
                            if remaining > 0:
                                await asyncio.sleep(remaining)
                                now = _time.monotonic()
                        self._last_ts = now

                        pts, time_base = await self.next_timestamp()
                        frame = VideoFrame.from_ndarray(
                            _np.zeros((self._height, self._width, 3), dtype=_np.uint8),
                            format="bgr24",
                        )
                        frame.pts = pts
                        frame.time_base = time_base
                        return frame

                self._black_video_track = BlackVideoTrack(width=self._target_video_width, height=self._target_video_height, fps=self._video_fps)
                try:
                    self._video_sender = self.pc.addTrack(self._black_video_track)
                    logger.info(
                        f"Negotiated sendonly video: sender={'yes' if self._video_sender else 'no'} "
                        f"black_track={'yes' if self._black_video_track else 'no'} fps={self._video_fps}"
                    )
                except Exception:
                    # If addTrack fails, disable video input gracefully
                    logger.warning("Failed to add black video track; video input disabled")
                    self._black_video_track = None
                    self._video_sender = None

            # Create offer and exchange SDP
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            answer_sdp = await self._exchange_sdp(offer.sdp)
            if not answer_sdp:
                raise RuntimeError("Failed to exchange SDP")

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

            # (SDP diagnostics removed)

            logger.info("WebRTC connection established, waiting for session creation")

            # Wait for session.created event, but don't hard-fail if it doesn't arrive
            try:
                await asyncio.wait_for(self.session_created_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("session.created not received within timeout; proceeding")

            # Update session configuration
            await self._update_session()

            self._running = True
            logger.info("OpenAI Realtime session started successfully")

            # Start outbound video RTP stats logging if video is enabled
            if self.enable_video_input:
                asyncio.create_task(self._log_outbound_video_stats())
                # Start periodic snapshot saver of the forwarded frame for verification (opt-in)
                if self._save_snapshots_enabled:
                    try:
                        os.makedirs(self._snapshot_dir, exist_ok=True)
                        logger.info(f"snapshot.dir.ready origin=SFU->OpenAI cache dir={self._snapshot_dir}")
                    except OSError as e:
                        logger.warning(f"Failed to create snapshot directory: {e}")
                    self._snapshot_task = asyncio.create_task(self._periodic_snapshot_saver())

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    async def _stop_session(self):
        """Stop the WebRTC session."""
        self._running = False

        try:
            if self._mic_track is not None:
                with contextlib.suppress(Exception):
                    self._mic_track.stop()
                self._mic_track = None
            if self.dc:
                self.dc.close()
                self.dc = None

            await self.pc.close()
            logger.info("OpenAI Realtime session stopped")

        except Exception as e:
            logger.error(f"Error stopping session: {e}")
        finally:
            # Close shared HTTP client
            try:
                if self._http is not None:
                    await self._http.aclose()
            except Exception:
                pass
            self._http = None
            # Stop any video tracks
            try:
                if self._forward_video_track is not None:
                    with contextlib.suppress(Exception):
                        self._forward_video_track.stop()  # type: ignore[attr-defined]
                if self._black_video_track is not None:
                    with contextlib.suppress(Exception):
                        self._black_video_track.stop()  # type: ignore[attr-defined]
            except Exception:
                pass
            # Stop snapshot task
            try:
                if self._snapshot_task is not None and not self._snapshot_task.done():
                    self._snapshot_task.cancel()
            except Exception:
                pass

    async def _update_session(self):
        """Update session configuration after creation."""
        event = {
            "type": "session.update",
            "session": {
                "instructions": self.system_instructions
                or "You are a helpful assistant.",
                "voice": self.voice,
                "input_audio_transcription": {"model": "gpt-4o-transcribe"},
                "turn_detection": {
                    "type": "semantic_vad" if self.turn_detection else None,
                    "eagerness": "auto",
                    "create_response": True,
                    "interrupt_response": True,
                }
                if self.turn_detection
                else None,
                "tools": [],
                "tool_choice": "auto",
                "temperature": 0.8,
                "max_response_output_tokens": 4096,
            },
        }
        await self._send_event(event)

    async def _handle_event(self, event: dict):
        """Handle events from OpenAI."""
        event_type = event.get("type")

        # Prominent logging for any event whose type contains 'video'
        try:
            if isinstance(event_type, str) and "video" in event_type.lower():
                banner = "\n" + ("🟪" * 40)
                logger.warning(f"{banner}\n🎥 OPENAI VIDEO EVENT: type={event_type}\n{banner}")
        except Exception:
            pass

        if event_type == "session.created":
            self.openai_session_id = event["session"]["id"]
            self.session_created_event.set()

        elif event_type == "response.created":
            logger.info(
                "response.created",
                extra={
                    "response_id": event.get("response", {}).get("id") or event.get("id"),
                },
            )

        elif event_type == "response.output_text.delta":
            # Streaming text delta from assistant (verbose only)
            if os.getenv("OPENAI_REALTIME_DEBUG"):
                delta = event.get("delta") or event.get("text")
                if delta:
                    print("[OpenAI Response Δ]", str(delta)[:200])

        elif event_type == "response.completed":
            rid = event.get("response", {}).get("id") or event.get("response_id") or event.get("id")
            logger.info("response.completed", extra={"response_id": rid})

        elif event_type == "conversation.item.created":
            # Item created in the conversation (user or assistant)
            item = event.get("item", {}) or {}
            content = item.get("content") or []
            content_types = [c.get("type") for c in content if isinstance(c, dict) and c.get("type")]
            logger.info(
                "conversation.item.created",
                extra={
                    "item_id": item.get("id"),
                    "role": item.get("role"),
                    "content_types": content_types,
                },
            )

        elif event_type == "response.audio.delta":
            # Ignore data-channel audio when using media track
            pass

        elif event_type == "response.audio_transcript.done":
            # Transcript of what the assistant said
            transcript = event.get("transcript")
            logger.info(f"Assistant: {transcript}")

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # Transcript of what the user said
            transcript = event.get("transcript")
            logger.info(f"User: {transcript}")

        elif event_type == "error":
            error = event.get("error", {})
            logger.error(f"OpenAI error: {error}")
            try:
                # On video-related errors, dump one frame snapshot stats if available
                msg = str(error)
                if "video" in msg.lower() and self._last_frame_rgb is not None:
                    mean = float(self._last_frame_rgb.mean())
                    h, w, *_ = self._last_frame_rgb.shape
                    logger.warning(
                        f"Latest forwarded frame stats: mean={mean:.2f} size={w}x{h} age={max(0.0, asyncio.get_event_loop().time() - self._last_frame_ts):.2f}s"
                    )
            except Exception:
                pass

        # Notify event callbacks
        for callback in self._event_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def _process_audio_track(self, track: MediaStreamTrack):
        """Process incoming audio track from OpenAI."""
        logger.info("Starting to process audio track from OpenAI (expect PCM frames)")
        print(
            "[OpenAI Realtime] Begin processing remote audio track",
            {
                "id": getattr(track, "id", None),
                "kind": getattr(track, "kind", None),
                "class": track.__class__.__name__,
                "readyState": getattr(track, "readyState", None),
            },
        )

        try:
            # Wait briefly for _running to become True in case on_track fired early
            if not self._running:
                for _ in range(50):  # up to ~5 seconds total
                    if self._running:
                        break
                    await asyncio.sleep(0.1)

            last_ready_state = getattr(track, "readyState", None)
            while self._running:
                try:
                    current_ready_state = getattr(track, "readyState", None)
                    if current_ready_state != last_ready_state:
                        print(
                            f"[OpenAI Realtime] Track readyState changed: {last_ready_state} -> {current_ready_state}"
                        )
                        last_ready_state = current_ready_state
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)

                    # Convert audio frame to mono int16 and resample to 48kHz (match WebRTC)
                    samples = frame.to_ndarray()
                    # Downmix stereo/planar to mono if needed (axis 0 = channels)
                    if samples.ndim == 2 and samples.shape[0] > 1:
                        samples = samples.mean(axis=0)
                    # Normalize dtype to int16
                    if samples.dtype != np.int16:
                        samples = (samples * 32767).astype(np.int16)
                    # OpenAI media track typically carries 24k PCM over data channel or 48k via WebRTC.
                    # We will upsample to 48000 Hz for publication to SFU.
                    in_rate = getattr(track, "sample_rate", None) or getattr(frame, "sample_rate", 48000) or 48000
                    if in_rate != 48000:
                        samples = resample_audio(samples, in_rate, 48000).astype(np.int16)
                    audio_bytes = samples.tobytes()

                    for callback in self._audio_callbacks:
                        await callback(audio_bytes)

                except asyncio.TimeoutError:
                    logger.debug("rx audio: no frame within 1s")
                    continue
                except MediaStreamError:
                    logger.info("Media stream ended")
                    print("[OpenAI Realtime] Remote audio media stream ended")
                    break

        except Exception as e:
            logger.error(f"Error processing audio track: {e}")
            try:
                print("[OpenAI Realtime] Error processing audio track:", e)
            except Exception:
                pass

    async def _log_inbound_audio_stats(self):
        """Periodically log inbound audio RTP stats if available."""
        while self._running:
            try:
                stats = await self.pc.getStats()
                for report in stats.values():
                    if (
                        report.type == "inbound-rtp"
                        and getattr(report, "kind", None) == "audio"
                    ):
                        bytes_rcv = getattr(report, "bytesReceived", None)
                        packs_rcv = getattr(report, "packetsReceived", None)
                        jitter = getattr(report, "jitter", None)
                        logger.info(
                            f"inbound audio stats: bytes={bytes_rcv} packets={packs_rcv} jitter={jitter}"
                        )
            except Exception:
                pass
            await asyncio.sleep(3.0)

    async def _log_outbound_video_stats(self):
        """Periodically log outbound video RTP stats if available."""
        while self._running and self.enable_video_input:
            try:
                stats = await self.pc.getStats()
                # Build a codec lookup by id for richer logging
                codecs = {rid: r for rid, r in stats.items() if getattr(r, "type", None) == "codec"}
                for report in stats.values():
                    if (
                        report.type == "outbound-rtp"
                        and getattr(report, "kind", None) == "video"
                    ):
                        bytes_sent = int(getattr(report, "bytesSent", 0) or 0)
                        frames_sent = getattr(report, "framesSent", None)
                        frames_sent = int(frames_sent or 0) if frames_sent is not None else None
                        packets_sent = int(getattr(report, "packetsSent", 0) or 0)
                        frame_w = getattr(report, "frameWidth", None)
                        frame_h = getattr(report, "frameHeight", None)
                        codec_id = getattr(report, "codecId", None)
                        codec = codecs.get(codec_id) if codec_id else None
                        mime = getattr(codec, "mimeType", None) if codec else None
                        # Include diffs to see whether content is actually flowing
                        prev = self._out_video_prev
                        d_bytes = bytes_sent - int(prev.get("bytes", 0))
                        d_pkts = packets_sent - int(prev.get("pkts", 0))
                        d_frames = (frames_sent - int(prev.get("frames", 0))) if frames_sent is not None else None
                        parts = [
                            f"bytes={bytes_sent} (+{d_bytes})",
                            f"packets={packets_sent} (+{d_pkts})",
                        ]
                        if d_frames is not None:
                            parts.append(f"frames={frames_sent} (+{d_frames})")
                        if frame_w and frame_h:
                            parts.append(f"size={frame_w}x{frame_h}")
                        if mime:
                            parts.append(f"codec={mime}")
                        logger.info("outbound video stats: " + " ".join(parts))
                        self._out_video_prev = {"bytes": bytes_sent, "pkts": packets_sent, "frames": frames_sent or 0}
                        # Stall detection: warn if frames not increasing for multiple intervals
                        if d_frames is not None and d_frames <= 0:
                            self._video_stall_count += 1
                            if self._video_stall_count >= 3:
                                logger.warning("outbound video appears stalled (no new frames for ~%ss)", 3 * 3)
                        else:
                            self._video_stall_count = 0
            except Exception:
                pass
            await asyncio.sleep(3.0)

    async def _periodic_snapshot_saver(self):
        """Save the last forwarded frame to disk every N seconds if available."""
        while self._running and self.enable_video_input and self._save_snapshots_enabled:
            try:
                await asyncio.sleep(self._snapshot_interval_sec)
                if self._last_frame_rgb is None:
                    logger.debug("snapshot: no last_frame_rgb yet")
                    continue
                rgb = self._last_frame_rgb
                # Convert to PNG and write
                try:
                    ts = int(time.time())
                    path = os.path.join(self._snapshot_dir, f"frame_{ts}.png")
                    await asyncio.to_thread(self._save_png, rgb, path)
                    logger.info(f"snapshot.saved (origin=SFU->OpenAI forward cache) path={path} ts={ts}")
                except Exception as e:
                    # Fallback: save raw RGB as .npy if Pillow not available or save failed
                    try:
                        ts = int(time.time())
                        path = os.path.join(self._snapshot_dir, f"frame_{ts}.npy")
                        await asyncio.to_thread(np.save, path, rgb)
                        logger.warning(f"snapshot.saved.raw (origin=SFU->OpenAI forward cache) path={path} ts={ts} error={e}")
                    except Exception as e2:
                        logger.debug(f"snapshot save failed (both PNG and NPY): {e2}")
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def __aiter__(self) -> AsyncIterator[dict]:
        """Iterate over events from OpenAI."""
        event_queue = asyncio.Queue()

        async def queue_events(event):
            await event_queue.put(event)

        self.on_event(queue_events)

        while self._running:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    # ---- Video input control ----
    async def start_video_sender(self, source_track: MediaStreamTrack, fps: Optional[int] = None) -> None:
        """Swap the negotiated dummy video to a forwarding track from source_track.

        Requires video to have been pre-negotiated (enable_video_input=True).
        """
        try:
            if not self.enable_video_input:
                raise RuntimeError("Video input not enabled for this session")
            # If already forwarding this exact source, skip
            if self._forward_video_track is not None and getattr(self._forward_video_track, "_source", None) is source_track:
                logger.info("start_video_sender: already forwarding this source; skipping")
                return

            # Find or reuse the active video sender to avoid repeated scans
            video_sender = self._active_video_sender
            if video_sender is None:
                for transceiver in self.pc.getTransceivers():
                    if getattr(transceiver, "kind", None) == "video" and getattr(transceiver, "sender", None):
                        video_sender = transceiver.sender
                        self._active_video_sender = video_sender
                        logger.warning(
                            f"start_video_sender: using transceiver kind=video dir={getattr(transceiver, 'direction', None)}"
                        )
                        break
            if video_sender is None:
                # Fallback to negotiated sender if present
                video_sender = self._video_sender
            if video_sender is None:
                raise RuntimeError("No active video sender found for replaceTrack")

            from aiortc.mediastreams import VideoStreamTrack as _VideoStreamTrack

            class ForwardingVideoTrack(_VideoStreamTrack):
                kind = "video"

                def __init__(self, source: MediaStreamTrack, fps_limit: int, outer_conn: "RealtimeConnection"):
                    super().__init__()
                    self._source = source
                    self._fps_limit = max(1, int(fps_limit))
                    self._last_ts = None
                    self._outer = outer_conn

                async def recv(self):
                    import time as _time

                    # Receive source frame first
                    frame: VideoFrame = await self._source.recv()
                    # Scale down frame to target resolution if larger
                    tgt_w = int(self._outer._target_video_width)
                    tgt_h = int(self._outer._target_video_height)
                    if tgt_w > 0 and tgt_h > 0 and (frame.width > tgt_w or frame.height > tgt_h):
                        scale = min(tgt_w / float(frame.width), tgt_h / float(frame.height))
                        new_w = max(1, int(frame.width * scale))
                        new_h = max(1, int(frame.height * scale))
                        try:
                            frame = frame.reformat(width=new_w, height=new_h)
                        except Exception:
                            # Keep original frame if resize unsupported
                            logger.warning("Failed to resize video frame; using original size")

                    # Throttle output to fps_limit
                    interval = 1.0 / float(self._fps_limit)
                    now = _time.monotonic()
                    if self._last_ts is not None:
                        remaining = (self._last_ts + interval) - now
                        if remaining > 0:
                            await asyncio.sleep(remaining)
                            now = _time.monotonic()
                    self._last_ts = now

                    # Stamp pts/time_base for the forwarded frame
                    pts, time_base = await self.next_timestamp()
                    frame.pts = pts
                    frame.time_base = time_base

                    # Optional: cache RGB for diagnostics/snapshots; avoid per-frame conversion unless enabled
                    if self._outer._video_debug_enabled or self._outer._save_snapshots_enabled:
                        arr = frame.to_ndarray(format="rgb24")
                        # Cache the last forwarded RGB frame for snapshotting
                        self._outer._last_frame_rgb = arr
                        self._outer._last_frame_ts = _time.monotonic()
                        if self._outer._video_debug_enabled:
                            try:
                                logger.debug(
                                    "video.forward: stage=SFU->OpenAI ts=%.3f pts=%s size=%sx%s fps_limit=%s",
                                    _time.time(),
                                    pts,
                                    arr.shape[1],
                                    arr.shape[0],
                                    self._fps_limit,
                                )
                                # Lightweight content fingerprint and stats (first frame only)
                                if getattr(self, "_debugged", False) is not True:
                                    mean = float(arr.mean()) if arr.size else 0.0
                                    vmin = float(arr.min()) if arr.size else 0.0
                                    vmax = float(arr.max()) if arr.size else 0.0
                                    h, w, c = arr.shape
                                    digest = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
                                    logger.debug(
                                        f"Forwarding video frame: sha256={digest} size={w}x{h} chans={c} mean={mean:.2f} min={vmin:.0f} max={vmax:.0f}"
                                    )
                                    # Save a one-off immediate snapshot for debugging
                                    if not self._outer._saved_first_snapshot:
                                        asyncio.create_task(self._outer._save_snapshot_now(arr, label="first_frame"))
                                        self._outer._saved_first_snapshot = True
                                    self._debugged = True
                            except Exception:
                                pass

                    return frame

            _fps = max(1, int(fps)) if fps is not None else self._video_fps
            # Pass outer connection reference into the forwarding track for snapshots
            self._forward_video_track = ForwardingVideoTrack(source_track, _fps, self)
            logger.warning(
                "start_video_sender: prepared ForwardingVideoTrack fps=%s source_kind=%s source_cls=%s",
                _fps,
                getattr(source_track, "kind", None),
                source_track.__class__.__name__,
            )
            # Ensure we are connected before swapping, to avoid early no-op
            tries = 0
            while self._last_conn_state not in ("connected", "completed") and tries < 50:
                await asyncio.sleep(0.1)
                tries += 1
            # replaceTrack may be synchronous in some aiortc versions
            _res = video_sender.replaceTrack(self._forward_video_track)
            if asyncio.iscoroutine(_res):
                logger.warning("start_video_sender: replaceTrack returned coroutine; awaiting")
                await _res
            else:
                logger.warning("start_video_sender: replaceTrack returned %s (not awaited)", type(_res).__name__)
            # Re-apply constraints after swap
            await self._apply_video_sender_constraints(video_sender)
            track_info = {
                "sender_has_track": bool(getattr(video_sender, "track", None)),
                "track_class": getattr(getattr(video_sender, "track", None), "__class__", type(None)).__name__,
            }
            logger.warning("Video sender switched to forwarding track (fps=%s) %s", _fps, track_info)

            # Kick once more after 1s to ensure downstream latched onto the new track
            async def _ensure_swap():
                await asyncio.sleep(1)
                _again = video_sender.replaceTrack(self._forward_video_track)
                if asyncio.iscoroutine(_again):
                    await _again
                logger.warning("Video sender reasserted forwarding track after delay")

            # Verify outbound stats shortly after swap
            async def _verify_outbound():
                try:
                    await asyncio.sleep(2)
                    stats = await self.pc.getStats()
                    frames = None
                    width = None
                    height = None
                    for r in stats.values():
                        if getattr(r, "type", None) == "outbound-rtp" and getattr(r, "kind", None) == "video":
                            frames = getattr(r, "framesSent", None)
                            width = getattr(r, "frameWidth", None)
                            height = getattr(r, "frameHeight", None)
                            break
                    logger.info("post-replaceTrack check: framesSent=%s size=%sx%s", frames, width, height)
                except Exception as e:
                    logger.debug(f"post-replaceTrack stats check failed: {e}")

            if not getattr(self, "_video_sender_started", False):
                asyncio.create_task(_ensure_swap())
                asyncio.create_task(_verify_outbound())
                self._video_sender_started = True
        except Exception as e:
            logger.error(f"Failed to start video sender: {e}")
            raise

    async def _apply_video_sender_constraints(self, sender) -> None:
        """Best-effort attempt to apply encoding constraints to the video sender.

        Works across aiortc versions by probing for setParameters or set_parameters.
        Currently only applies maxFramerate to avoid adding latency via encoder buffering.
        """
        params = None
        # Try modern API first
        if hasattr(sender, "getParameters"):
            params = sender.getParameters()
        # Fallback older naming
        if params is None and hasattr(sender, "get_parameters"):
            params = sender.get_parameters()

        if params is None:
            return

        # Ensure encodings exists
        encs = getattr(params, "encodings", None)
        if encs is None:
            params.encodings = [{}]
            encs = params.encodings

        if encs is not None and len(encs) > 0:
            enc = encs[0]
            # Apply only framerate; leave bitrate to defaults negotiated with SFU/provider
            enc["maxFramerate"] = int(self._video_fps)

        # Apply back using whichever method exists
        if hasattr(sender, "setParameters"):
            await sender.setParameters(params)  # type: ignore[func-returns-value]
            return
        if hasattr(sender, "set_parameters"):
            await sender.set_parameters(params)  # type: ignore[func-returns-value]
            return

    async def stop_video_sender(self) -> None:
        """Restore the black video track (or disable if unavailable)."""
        try:
            if self._video_sender and self._black_video_track:
                _res = self._video_sender.replaceTrack(self._black_video_track)
                if asyncio.iscoroutine(_res):
                    await _res
                logger.info("Video sender reverted to black video track")
            # Stop forwarding track if it exists
            if self._forward_video_track is not None:
                with contextlib.suppress(Exception):
                    self._forward_video_track.stop()  # type: ignore[attr-defined]
            self._forward_video_track = None
        except Exception as e:
            logger.error(f"Failed to stop video sender: {e}")


    async def _save_snapshot_now(self, rgb, label: str = "immediate") -> None:
        """Immediately save a snapshot from provided RGB ndarray for diagnostics."""
        try:
            os.makedirs(self._snapshot_dir, exist_ok=True)
        except Exception:
            pass
        ts = int(time.time())
        # Try PNG first
        try:
            path = os.path.join(self._snapshot_dir, f"{label}_{ts}.png")
            await asyncio.to_thread(self._save_png, rgb, path)
            logger.info(f"Saved immediate snapshot: {path}")
            return
        except Exception as e:
            try:
                path = os.path.join(self._snapshot_dir, f"{label}_{ts}.npy")
                await asyncio.to_thread(np.save, path, rgb)
                logger.warning(f"Immediate snapshot PNG failed ({e}); saved raw RGB to: {path}")
            except Exception as e2:
                logger.debug(f"Immediate snapshot save failed (both PNG and NPY): {e2}")

    def _save_png(self, rgb, path: str) -> None:
        from PIL import Image  # type: ignore
        img = Image.fromarray(rgb, mode="RGB")
        img.save(path, format="PNG")


class Realtime(realtime.Realtime):
    """
    OpenAI Realtime API with Speech-to-Speech capabilities.

    This Realtime implementation enables real-time, bidirectional audio conversations with GPT-4
    using WebRTC. It supports both text and audio modalities.

    The agent requires that we standardize:
    - Realtime connection management via connect() method
    - Audio forwarding via send_audio_pcm()
    - Response handling (though mostly happens via audio)

    Events:
    - connected: When WebRTC connection is established
    - disconnected: When connection is closed
    - transcript: When a transcript is available (user or assistant)
    - response: When a response is complete
    - audio_output: When audio is received from OpenAI
    - error: When an error occurs
    """

    def __init__(
        self,
        model: str = "gpt-realtime",
        api_key: Optional[str] = None,
        voice: str = "alloy",
        turn_detection: bool = True,
        instructions: Optional[str] = None,
        client: Optional[Any] = None,  # For compatibility with base class
        *,
        barge_in: bool = True,
        activity_threshold: int = 3000,
        silence_timeout_ms: int = 1000,
        enable_video_input: bool = False,
        video_fps: int = 1,
        # Client-configurable video parameters
        video_width: int = 1280,
        video_height: int = 720,
        video_debug_enabled: bool = False,
        save_snapshots_enabled: bool = False,
        snapshot_interval_sec: float = 3.0,
    ):
        """Initialize OpenAI Realtime Realtime.

        Args:
            model: The model to use (default: gpt-4o-realtime-preview-2024-12-17)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            turn_detection: Enable automatic turn detection
            instructions: System instructions for the assistant
            client: Not used, kept for compatibility
        """
        super().__init__(
            provider_name="openai-realtime",
            model=model,
            instructions=instructions,
            voice=voice,
            provider_config={
                "turn_detection": turn_detection,
            },
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.turn_detection = turn_detection
        self.system_prompt = instructions
        self.realtime = True  # This is a Realtime-capable LLM
        self._connection: Optional[RealtimeConnection] = None
        self.conversation = None  # For compatibility
        self._connection_lock = asyncio.Lock()
        # Local playback gating to allow barge-in style interruption if desired
        self._playback_enabled: bool = True
        # Barge-in controls (optional, local playback only; provider VAD still used upstream)
        self._barge_in_enabled: bool = barge_in
        self._activity_threshold: int = activity_threshold
        self._silence_timeout_ms: int = silence_timeout_ms
        self._user_speaking: bool = False
        self._eos_timer_task: Optional[asyncio.Task] = None
        # Video input enablement (pre-negotiate sendonly video with a black track)
        self.enable_video_input: bool = enable_video_input
        self.video_fps: int = max(1, int(video_fps))
        self.openai_session_id: Optional[str] = None
        # Exposed video parameters to pass into the connection
        self.video_width: int = max(1, int(video_width))
        self.video_height: int = max(1, int(video_height))
        self.video_debug_enabled: bool = bool(video_debug_enabled)
        self.save_snapshots_enabled: bool = bool(save_snapshots_enabled)
        self.snapshot_interval_sec: float = max(0.5, float(snapshot_interval_sec))

        # Auto-start the realtime connection so Agent.wait_until_ready() does not hang
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._ensure_connection())
        except RuntimeError:
            # Not in an event loop; connection will be created on first use
            pass

    async def _ensure_connection(self):
        """Ensure we have an active connection, creating one if needed."""
        async with self._connection_lock:
            if not self._connection or not self._is_connected:
                await self._create_connection()

    async def _create_connection(self):
        """Create and setup the WebRTC connection."""
        try:
            self._connection = RealtimeConnection(
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
                save_snapshots_enabled=self.save_snapshots_enabled,
                snapshot_interval_sec=self.snapshot_interval_sec,
            )

            # Register event handlers
            self._connection.on_event(self._handle_openai_event)
            self._connection.on_audio(self._handle_audio_output)

            # Start the connection
            await self._connection._start_session()

            # Mark as connected before emitting event
            self._is_connected = True

            # Emit connected event using Realtime helper
            # Manually emit connected event (include provider) and mark ready
            self._is_connected = True
            self._ready_event.set()
            event = RealtimeConnectedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                provider="openai",
                session_config={
                    "model": self.model,
                    "voice": self.voice,
                    "turn_detection": self.turn_detection,
                },
                capabilities=["text", "audio"],
            )
            register_global_event(event)
            self.emit("connected", event)

            # Prime publisher audio pipeline with brief silence to seed timestamps
            output_track = getattr(self, "output_track", None)
            if output_track is not None:
                # 100ms of silence at 48kHz mono s16 -> 4800 samples -> 9600 bytes
                await output_track.write(b"\x00" * 9600)

        except Exception as e:
            # Emit error event using Realtime helper
            self._emit_error_event(e, "connection")
            raise

    async def _handle_openai_event(self, event: dict):
        """Handle events from OpenAI and emit appropriate Realtime events."""
        event_type = event.get("type")

        if event_type == "session.created":
            self.openai_session_id = event["session"]["id"]
            logger.info(
                f"OpenAI session.created openai_session_id={self.openai_session_id} local_session_id={self.session_id}"
            )

        if event_type == "response.audio_transcript.done":
            # Assistant's transcript
            transcript = event.get("transcript", "")
            if transcript:
                # Emit transcript event using Realtime helper
                self._emit_transcript_event(
                    text=transcript,
                    user_metadata={"role": "assistant", "source": "openai"},
                )

                # Also emit response event using Realtime helper
                self._emit_response_event(
                    text=transcript,
                    response_id=event.get("response_id"),
                    is_complete=True,
                    conversation_item_id=event.get("item_id"),
                )

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # User's transcript
            transcript = event.get("transcript", "")
            if transcript:
                # Emit transcript event using Realtime helper
                self._emit_transcript_event(
                    text=transcript,
                    user_metadata={"role": "user", "source": "openai"},
                )

        elif event_type == "error":
            error = event.get("error", {})
            # Emit error event using Realtime helper
            self._emit_error_event(
                error=Exception(error.get("message", "Unknown error")),
                context=f"openai_event: {error.get('code', 'unknown')}",
            )

    async def _handle_audio_output(self, audio_bytes: bytes):
        """Handle audio output from OpenAI."""
        # Emit audio output event only if someone is listening to avoid overhead
        listeners_fn = getattr(self, "listeners", None)
        has_listeners = bool(listeners_fn("audio_output")) if callable(listeners_fn) else False
        if has_listeners:
            self._emit_audio_output_event(
                audio_data=audio_bytes,
                sample_rate=48000,
            )
        # Also push audio to the published output track so remote participants hear it
        output_track = getattr(self, "output_track", None)
        if output_track is not None and self._playback_enabled:
            try:
                await output_track.write(audio_bytes)
            except Exception as e:
                logger.debug(f"Failed to write audio to output track: {e}")

    @asynccontextmanager
    async def connect(self, call: Call, agent_user_id: str):
        """Create a connection context for the OpenAI Realtime session.

        This method is called by the Agent when joining a call in Realtime mode.
        """
        logger.info(f"Connecting OpenAI Realtime for call {call.id}")

        # Ensure connection is created
        await self._ensure_connection()

        try:
            yield self._connection
        finally:
            # Don't close connection here - it persists across calls
            pass

    async def send_audio_pcm(self, pcm_data, target_rate: int = 48000):
        """Send PCM audio data to OpenAI (awaitable)."""
        await self._send_audio_pcm_async(pcm_data, target_rate)

    async def _send_audio_pcm_async(self, pcm_data, target_rate: int):
        """Async implementation of send_audio_pcm."""
        # Ensure connection exists
        await self._ensure_connection()

        if not self._connection:
            logger.warning("No active OpenAI connection")
            return

        # Extract numpy int16 array from PCM data and resample to 48000 Hz for RTP
        audio_bytes = None
        samples = getattr(pcm_data, "samples", None)
        if samples is not None:
            if not isinstance(samples, (bytes, bytearray)):
                import numpy as _np

                arr = _np.asarray(samples)
                if arr.size == 0:
                    return
                if arr.dtype != _np.int16:
                    arr = arr.astype(_np.int16)
                # Use source sample rate; avoid Python-side resampling
                src_rate = getattr(pcm_data, "sample_rate", target_rate)
                if not src_rate:
                    src_rate = 48000
                if arr.size == 0:
                    return
                audio_bytes = arr.tobytes()
            else:
                # Raw bytes; assume already int16 48kHz; drop too-small frames (<20ms)
                if len(samples) < 1920:
                    return
                audio_bytes = bytes(samples)
        elif isinstance(pcm_data, bytes):
            # Expect 48kHz mono s16; drop too-small frames (<20ms)
            if len(pcm_data) < 1920:
                return
            audio_bytes = pcm_data

        if audio_bytes:
            # Emit input event using Realtime helper (report source rate)
            src_rate_emit = int(getattr(pcm_data, "sample_rate", 48000))
            self._emit_audio_input_event(
                audio_data=audio_bytes,
                sample_rate=src_rate_emit,
            )

            # Compute 20ms frame size at source sample rate
            src_rate = src_rate_emit
            samples_per_frame = int(0.02 * src_rate)
            frame_bytes = samples_per_frame * 2  # int16 mono
            if frame_bytes <= 0:
                frame_bytes = 1920  # fallback to 48k 20ms

            total_frames = max(1, (len(audio_bytes) + frame_bytes - 1) // frame_bytes)
            max_forward_frames = 3
            start_frame = max(0, total_frames - max_forward_frames)
            start_offset = start_frame * frame_bytes

            # Chunk only the recent portion and enqueue with backpressure, carrying sample rate
            for i in range(start_offset, len(audio_bytes), frame_bytes):
                chunk = audio_bytes[i : i + frame_bytes]
                if not chunk:
                    continue
                if len(chunk) < frame_bytes:
                    chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
                # Optional local barge-in: detect activity and interrupt playback immediately
                if self._barge_in_enabled:
                    try:
                        arr = np.frombuffer(chunk, dtype=np.int16)
                        if arr.size:
                            energy = float(np.mean(np.abs(arr)))
                            is_active = energy > float(self._activity_threshold)
                            if is_active and not self._user_speaking:
                                await self.interrupt_playback()
                                self._user_speaking = True
                            if is_active:
                                # Restart end-of-speech timer
                                if self._eos_timer_task and not self._eos_timer_task.done():
                                    self._eos_timer_task.cancel()
                                self._eos_timer_task = asyncio.create_task(self._silence_timeout_task())
                    except Exception:
                        # Do not let barge-in logic disrupt audio sending
                        pass
                await self._connection.send_audio(chunk, sample_rate=src_rate)

    async def send_text(self, text: str):
        """Send a text message from the human side to the conversation."""
        await self._ensure_connection()
        # Emit user transcript event for UI mirroring
        self._emit_transcript_event(text=text, user_metadata={"role": "user"})
        # Ensure playback is enabled before expecting an assistant reply
        self._playback_enabled = True
        if not self._connection:
            raise RuntimeError("Failed to establish connection")
        await self._connection.send_text(text)

    async def simple_response(
        self,
        *,
        text: str,

        timeout: Optional[float] = 30.0,
    ):
        """Standardized single-turn response using base aggregation."""
        return await super().simple_response(
            text=text, timeout=timeout
        )

    async def create_response(self, *args, **kwargs):
        """Compatibility wrapper mapping to simple_response."""
        text = kwargs.get("input", args[0] if args else "")
        rt_resp = await self.simple_response(text=text)
        # Wrap into legacy LLMResponse for compatibility with older tests
        return LLMResponse(original=rt_resp.original, text=rt_resp.text)

    async def start_video_sender(self, track: MediaStreamTrack, fps: Optional[int] = None) -> None:
        """Start forwarding video frames upstream. Requires enable_video_input=True."""
        await self._ensure_connection()
        await self._connection.start_video_sender(track, fps if fps is not None else self.video_fps)

    async def stop_video_sender(self) -> None:
        """Stop forwarding and revert to the negotiated black track (if any)."""
        if self._connection:
            await self._connection.stop_video_sender()

    async def _close_impl(self):
        """Close the Realtime service and release resources."""
        # Cancel end-of-speech timer if running
        if self._eos_timer_task and not self._eos_timer_task.done():
            self._eos_timer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._eos_timer_task
        self._eos_timer_task = None
        if self._connection:
            try:
                await self._connection._stop_session()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None
        # Do not call parent close here; base .close() orchestrates lifecycle

    async def interrupt_playback(self) -> None:
        """Stop current playback immediately and clear queued audio chunks."""
        self._playback_enabled = False
        output_track = getattr(self, "output_track", None)
        if output_track is not None:
            flush_fn = getattr(output_track, "flush", None)
            if callable(flush_fn):
                with contextlib.suppress(Exception):
                    await flush_fn()

    def resume_playback(self) -> None:
        """Re-enable playback after an interruption."""
        self._playback_enabled = True

    async def _silence_timeout_task(self) -> None:
        try:
            await asyncio.sleep(self._silence_timeout_ms / 1000)
            self._user_speaking = False
            self.resume_playback()
        except asyncio.CancelledError:
            return
