"""OpenAI Realtime API integration with WebRTC."""

import asyncio
import json
import logging
import os
import traceback
from typing import Optional, List, Any, AsyncIterator
from contextlib import asynccontextmanager
import requests
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
    AudioStreamTrack,
)
from aiortc.contrib.media import MediaStreamTrack, MediaStreamError
import numpy as np
from av import AudioFrame
from fractions import Fraction

from getstream.video.call import Call
from getstream.audio.utils import resample_audio
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from stream_agents.core.llm import realtime
from stream_agents.core.llm.llm import LLMResponse
from stream_agents.core.processors import BaseProcessor
from stream_agents.core.events import (
    RealtimeConnectedEvent,
    register_global_event,
)


logger = logging.getLogger(__name__)


class RealtimeConnection:
    """Internal WebRTC connection handler for OpenAI Realtime API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        voice: str = "alloy",
        turn_detection: bool = True,
        system_instructions: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.turn_detection = turn_detection
        self.system_instructions = system_instructions

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
        try:
            while self._mic_queue.qsize() >= self._max_queue_frames:
                try:
                    _ = self._mic_queue.get_nowait()
                    self._frames_dropped += 1
                except Exception:
                    break
            # Store (bytes, sample_rate) tuple
            self._mic_queue.put_nowait((audio_data, sample_rate))
            if self._frames_dropped and (self._frames_dropped % 50 == 0):
                logger.debug(
                    f"mic_queue: dropped_frames={self._frames_dropped} size={self._mic_queue.qsize()}"
                )
        except Exception as e:
            logger.debug(f"mic_queue enqueue error: {e}")

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

    def _get_session_token(self) -> Optional[str]:
        """Get a session token from OpenAI."""
        try:
            # Use the correct endpoint for ephemeral keys
            url = "https://api.openai.com/v1/realtime/sessions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Session configuration with correct structure
            session_config = {
                "model": self.model,
                # Set voice at session creation so assistant will produce audio over media track
                "voice": self.voice,
            }

            response = requests.post(url, headers=headers, json=session_config)
            response.raise_for_status()

            data = response.json()
            logger.info("Successfully obtained OpenAI session token")

            # Extract the client secret value
            client_secret = data.get("client_secret")
            if isinstance(client_secret, dict):
                return client_secret.get("value")
            return client_secret

        except Exception as e:
            logger.error(f"Failed to get session token: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    def _exchange_sdp(self, sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI."""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/sdp",
            }

            # Use the correct endpoint format
            url = f"https://api.openai.com/v1/realtime?model={self.model}"

            response = requests.post(
                url,
                headers=headers,
                data=sdp,
            )

            response.raise_for_status()
            logger.info("SDP exchange successful")

            if response.text:
                return response.text
            else:
                logger.error("Received empty SDP response")
                return None

        except Exception as e:
            logger.error(f"SDP exchange failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return None

    async def _start_session(self):
        """Start the WebRTC session with OpenAI."""
        try:
            # Get session token
            self.token = self._get_session_token()
            if not self.token:
                raise RuntimeError("Failed to obtain session token")

            # Set up peer connection event handlers
            @self.pc.on("track")
            async def on_track(track):
                if track.kind == "audio":
                    try:
                        track_id = getattr(track, "id", "<no-id>")
                        logger.info(
                            f"on_track: remote audio track attached id={track_id}"
                        )
                    except Exception:
                        logger.info("on_track: remote audio track attached")
                    # Extra diagnostics for the audio track in debug mode
                    try:
                        print(
                            "[OpenAI Realtime] Remote audio track attached:",
                            {
                                "id": getattr(track, "id", None),
                                "kind": getattr(track, "kind", None),
                                "class": track.__class__.__name__,
                                "readyState": getattr(track, "readyState", None),
                            },
                        )
                    except Exception:
                        pass
                    asyncio.create_task(self._process_audio_track(track))
                else:
                    logger.info(f"on_track: non-audio track received kind={track.kind}")

            # Connection state diagnostics
            # Minimal connection state diagnostics
            @self.pc.on("connectionstatechange")
            async def on_conn_state_change():
                logger.info(f"pc.connectionState={self.pc.connectionState}")

            # Create data channel for events
            self.dc = self.pc.createDataChannel("oai-events")

            @self.dc.on("open")
            def on_open():
                logger.info("Data channel opened")

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
                        num_samples = int(0.02 * self._sample_rate)
                        samples = np.zeros((1, num_samples), dtype=np.int16)

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

            # Create offer and exchange SDP
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            answer_sdp = self._exchange_sdp(offer.sdp)
            if not answer_sdp:
                raise RuntimeError("Failed to exchange SDP")

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

            # (SDP diagnostics removed)

            logger.info("WebRTC connection established, waiting for session creation")

            # Wait for session.created event
            await asyncio.wait_for(self.session_created_event.wait(), timeout=10.0)

            # Update session configuration
            await self._update_session()

            self._running = True
            logger.info("OpenAI Realtime session started successfully")

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise

    async def _stop_session(self):
        """Stop the WebRTC session."""
        self._running = False

        try:
            if self._mic_track is not None:
                try:
                    self._mic_track.stop()
                except Exception:
                    pass
                self._mic_track = None
            if self.dc:
                self.dc.close()
                self.dc = None

            await self.pc.close()
            logger.info("OpenAI Realtime session stopped")

        except Exception as e:
            logger.error(f"Error stopping session: {e}")

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

        if event_type == "session.created":
            logger.info("Session created")
            self.session_created_event.set()

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

        # Notify event callbacks
        for callback in self._event_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    async def _process_audio_track(self, track: MediaStreamTrack):
        """Process incoming audio track from OpenAI."""
        logger.info("Starting to process audio track from OpenAI (expect PCM frames)")
        try:
            print(
                "[OpenAI Realtime] Begin processing remote audio track",
                {
                    "id": getattr(track, "id", None),
                    "kind": getattr(track, "kind", None),
                    "class": track.__class__.__name__,
                    "readyState": getattr(track, "readyState", None),
                },
            )
        except Exception:
            print("🔊 Error processing audio track")
            print(traceback.format_exc())
            pass

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
                    if hasattr(frame, "to_ndarray"):
                        samples = frame.to_ndarray()
                        # Downmix stereo/planar to mono if needed (axis 0 = channels)
                        if samples.ndim == 2 and samples.shape[0] > 1:
                            samples = samples.mean(axis=0)
                        # Normalize dtype to int16
                        if samples.dtype != np.int16:
                            samples = (samples * 32767).astype(np.int16)
                        # OpenAI media track typically carries 24k PCM over data channel or 48k via WebRTC.
                        # We will upsample to 48000 Hz for publication to SFU.
                        in_rate = getattr(frame, "sample_rate", 48000) or 48000
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
        try:
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
                    await asyncio.sleep(2.0)
                except Exception:
                    await asyncio.sleep(2.0)
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
        system_prompt: Optional[str] = None,
        client: Optional[Any] = None,  # For compatibility with base class
    ):
        """Initialize OpenAI Realtime Realtime.

        Args:
            model: The model to use (default: gpt-4o-realtime-preview-2024-12-17)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            turn_detection: Enable automatic turn detection
            system_prompt: System instructions for the assistant
            client: Not used, kept for compatibility
        """
        super().__init__(
            provider_name="openai-realtime",
            model=model,
            instructions=system_prompt,
            voice=voice,
            provider_config={
                "turn_detection": turn_detection,
            },
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self.turn_detection = turn_detection
        self.system_prompt = system_prompt
        self.realtime = True  # This is a Realtime-capable LLM
        self._connection: Optional[RealtimeConnection] = None
        self.conversation = None  # For compatibility
        self._connection_lock = asyncio.Lock()

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
            try:
                self._ready_event.set()
            except Exception:
                pass
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
            try:
                if hasattr(self, "output_track") and self.output_track is not None:
                    # 200ms of silence at 24kHz mono s16 -> 4800 samples -> 9600 bytes
                    await self.output_track.write(b"\x00" * 9600)
            except Exception:
                pass

        except Exception as e:
            # Emit error event using Realtime helper
            self._emit_error_event(e, "connection")
            raise

    async def _handle_openai_event(self, event: dict):
        """Handle events from OpenAI and emit appropriate Realtime events."""
        event_type = event.get("type")

        if event_type == "response.audio_transcript.done":
            # Assistant's transcript
            transcript = event.get("transcript", "")
            if transcript:
                # Emit transcript event using Realtime helper
                self._emit_transcript_event(
                    text=transcript,
                    is_user=False,
                    conversation_item_id=event.get("item_id"),
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
                    is_user=True,
                    conversation_item_id=event.get("item_id"),
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
        # Emit audio output event using Realtime helper
        self._emit_audio_output_event(
            audio_data=audio_bytes,
            sample_rate=48000,
        )
        # Also push audio to the published output track so remote participants hear it
        try:
            if hasattr(self, "output_track") and self.output_track is not None:
                # Write raw PCM bytes at 48kHz to published output track
                await self.output_track.write(audio_bytes)
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
        if hasattr(pcm_data, "samples"):
            samples = pcm_data.samples
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
            src_rate_emit = int(getattr(pcm_data, "sample_rate", 48000)) if hasattr(pcm_data, "sample_rate") else 48000
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
                await self._connection.send_audio(chunk, sample_rate=src_rate)

    async def send_text(self, text: str):
        """Send a text message from the human side to the conversation."""
        await self._ensure_connection()
        # Emit user transcript event for UI mirroring
        try:
            self._emit_transcript_event(text=text, is_user=True)
        except Exception:
            pass
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

    async def _close_impl(self):
        """Close the Realtime service and release resources."""
        if self._connection:
            try:
                await self._connection._stop_session()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None

        # Do not call parent close here; base .close() orchestrates lifecycle
