"""OpenAI Realtime API integration for Speech-to-Speech (STS) capabilities."""

import asyncio
import json
import logging
import os
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
import concurrent.futures

from getstream.video.call import Call
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant

from stream_agents.llm.sts import STS
from stream_agents.llm.llm import LLMResponse
from stream_agents.processors import BaseProcessor
from stream_agents.events import (
    STSConnectedEvent,
    register_global_event,
)


logger = logging.getLogger(__name__)


class OpenAIRealtimeConnection:
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
        self._audio_track: Optional[AudioStreamTrack] = None

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

    async def send_audio(self, audio_data: bytes):
        """Send audio data to OpenAI."""
        if not self.dc:
            logger.warning("Data channel not ready, cannot send audio")
            return

        # OpenAI expects audio to be sent via the data channel as events
        event = {
            "type": "input_audio_buffer.append",
            "audio": audio_data.hex(),  # Convert bytes to hex string
        }
        await self._send_event(event)

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
        await self._send_event({"type": "response.create"})

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
                "modalities": ["text", "audio"],  # Enable both text and audio
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
                    logger.info("Audio track received from OpenAI")
                    self._audio_track = track
                    asyncio.create_task(self._process_audio_track(track))

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

            # Add an audio track to send audio to OpenAI
            # Create a dummy audio source for the offer
            from aiortc.mediastreams import AudioStreamTrack

            class DummyAudioTrack(AudioStreamTrack):
                """Dummy audio track for SDP negotiation."""

                def __init__(self):
                    super().__init__()

                async def recv(self):
                    # This won't actually be called for sending audio
                    # We'll use the data channel for sending audio
                    await asyncio.sleep(0.1)
                    return None

            # Add the audio track to enable audio in SDP
            audio_track = DummyAudioTrack()
            self.pc.addTrack(audio_track)

            # Create offer and exchange SDP
            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            answer_sdp = self._exchange_sdp(offer.sdp)
            if not answer_sdp:
                raise RuntimeError("Failed to exchange SDP")

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)

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
                "modalities": ["text", "audio"],
                "instructions": self.system_instructions
                or "You are a helpful assistant.",
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad" if self.turn_detection else None,
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 200,
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
            # Audio data from OpenAI (base64 encoded)
            audio_b64 = event.get("delta")
            if audio_b64:
                # Decode base64 to bytes
                import base64

                audio_bytes = base64.b64decode(audio_b64)
                # Notify audio callbacks
                for callback in self._audio_callbacks:
                    await callback(audio_bytes)

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
        logger.info("Starting to process audio track from OpenAI")

        try:
            while self._running:
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)

                    # Convert audio frame to PCM bytes
                    # OpenAI sends 24kHz mono PCM16
                    if hasattr(frame, "to_ndarray"):
                        samples = frame.to_ndarray()
                        # Ensure int16
                        if samples.dtype != np.int16:
                            samples = (samples * 32767).astype(np.int16)
                        audio_bytes = samples.tobytes()

                        # Notify audio callbacks
                        for callback in self._audio_callbacks:
                            await callback(audio_bytes)

                except asyncio.TimeoutError:
                    continue
                except MediaStreamError:
                    logger.info("Media stream ended")
                    break

        except Exception as e:
            logger.error(f"Error processing audio track: {e}")

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


class OpenAIRealtimeLLM(STS):
    """
    OpenAI Realtime API with Speech-to-Speech capabilities.

    This STS implementation enables real-time, bidirectional audio conversations with GPT-4
    using WebRTC. It supports both text and audio modalities.

    The agent requires that we standardize:
    - STS connection management via connect() method
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
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        api_key: Optional[str] = None,
        voice: str = "alloy",
        turn_detection: bool = True,
        system_prompt: Optional[str] = None,
        client: Optional[Any] = None,  # For compatibility with base class
    ):
        """Initialize OpenAI Realtime STS.

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
        self.sts = True  # This is an STS-capable LLM
        self._connection: Optional[OpenAIRealtimeConnection] = None
        self.conversation = None  # For compatibility
        self._connection_lock = asyncio.Lock()

    async def _ensure_connection(self):
        """Ensure we have an active connection, creating one if needed."""
        async with self._connection_lock:
            if not self._connection or not self._is_connected:
                await self._create_connection()

    async def _create_connection(self):
        """Create and setup the WebRTC connection."""
        try:
            self._connection = OpenAIRealtimeConnection(
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

            # Emit connected event using STS helper
            # Note: STS base class stores the provider in self.provider_name
            event = STSConnectedEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                provider="openai",  # Set provider explicitly
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
            # Emit error event using STS helper
            self._emit_error_event(e, "connection")
            raise

    async def _handle_openai_event(self, event: dict):
        """Handle events from OpenAI and emit appropriate STS events."""
        event_type = event.get("type")

        if event_type == "response.audio_transcript.done":
            # Assistant's transcript
            transcript = event.get("transcript", "")
            if transcript:
                # Emit transcript event using STS helper
                self._emit_transcript_event(
                    text=transcript,
                    is_user=False,
                    conversation_item_id=event.get("item_id"),
                )

                # Also emit response event using STS helper
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
                # Emit transcript event using STS helper
                self._emit_transcript_event(
                    text=transcript,
                    is_user=True,
                    conversation_item_id=event.get("item_id"),
                )

        elif event_type == "error":
            error = event.get("error", {})
            # Emit error event using STS helper
            self._emit_error_event(
                error=Exception(error.get("message", "Unknown error")),
                context=f"openai_event: {error.get('code', 'unknown')}",
            )

    async def _handle_audio_output(self, audio_bytes: bytes):
        """Handle audio output from OpenAI."""
        # Emit audio output event using STS helper
        self._emit_audio_output_event(
            audio_data=audio_bytes,
            sample_rate=24000,  # OpenAI uses 24kHz
        )

    @asynccontextmanager
    async def connect(self, call: Call, agent_user_id: str):
        """Create a connection context for the OpenAI Realtime session.

        This method is called by the Agent when joining a call in STS mode.
        """
        logger.info(f"Connecting OpenAI Realtime for call {call.id}")

        # Ensure connection is created
        await self._ensure_connection()

        try:
            yield self._connection
        finally:
            # Don't close connection here - it persists across calls
            pass

    def send_audio_pcm(self, pcm_data, target_rate: int = 48000):
        """Send PCM audio data to OpenAI.

        This method is called by the Agent when audio is received from participants.
        Note: This is synchronous as required by the abstract method, but we create
        an async task to handle the actual sending.
        """
        # Create task to handle async operations
        asyncio.create_task(self._send_audio_pcm_async(pcm_data, target_rate))

    async def _send_audio_pcm_async(self, pcm_data, target_rate: int):
        """Async implementation of send_audio_pcm."""
        # Ensure connection exists
        await self._ensure_connection()

        if not self._connection:
            logger.warning("No active OpenAI connection")
            return

        # Extract audio bytes from PCM data
        audio_bytes = None
        if hasattr(pcm_data, "samples"):
            samples = pcm_data.samples
            if hasattr(samples, "tobytes"):
                audio_bytes = samples.tobytes()
            else:
                audio_bytes = samples.astype("int16").tobytes()
        elif isinstance(pcm_data, bytes):
            audio_bytes = pcm_data

        if audio_bytes:
            # Emit input event using STS helper
            self._emit_audio_input_event(
                audio_data=audio_bytes,
                sample_rate=target_rate,
            )

            # Send to OpenAI
            await self._connection.send_audio(audio_bytes)

    def simple_response(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponse:
        """Generate a response for text-based input.

        Note: In STS mode, responses are typically generated automatically
        through the audio conversation. This method sends text input and
        waits for a response.
        """
        # Run async implementation in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, use run_coroutine_threadsafe
                future = asyncio.run_coroutine_threadsafe(
                    self._simple_response_async(text, processors, participant), loop
                )
                # Wait for the result with a timeout
                return future.result(timeout=15.0)
            else:
                # If no loop is running, we can run directly
                return asyncio.run(
                    self._simple_response_async(text, processors, participant)
                )
        except concurrent.futures.TimeoutError:
            logger.error("Timeout in simple_response")
            return LLMResponse(
                original={"text": text, "error": "timeout"}, text="[Response timeout]"
            )

    async def _simple_response_async(
        self,
        text: str,
        processors: Optional[List[BaseProcessor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponse:
        """Async implementation of simple_response."""
        # Ensure connection exists
        await self._ensure_connection()

        if not self._connection:
            raise RuntimeError("Failed to establish connection")

        # Track responses
        responses = []
        response_event = asyncio.Event()

        def on_response(event):
            if hasattr(event, "text") and event.text:
                responses.append(event.text)
                response_event.set()

        # Temporarily subscribe to responses
        self.on("response", on_response)

        try:
            # Send text
            await self._connection.send_text(text)

            # Wait for response (with timeout)
            await asyncio.wait_for(response_event.wait(), timeout=10.0)

            response_text = " ".join(responses) if responses else "[No response]"
            return LLMResponse(
                original={"text": text, "response": response_text}, text=response_text
            )

        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for response")
            return LLMResponse(
                original={"text": text, "error": "timeout"}, text="[Response timeout]"
            )
        finally:
            # Unsubscribe
            self.remove_listener("response", on_response)

    async def create_response(self, *args, **kwargs) -> LLMResponse:
        """Native OpenAI Realtime API method.

        This maps to simple_response for compatibility.
        """
        # Extract text from kwargs
        text = kwargs.get("input", args[0] if args else "")
        return await self._simple_response_async(text)

    async def close(self):
        """Close the STS service and release resources."""
        if self._connection:
            try:
                await self._connection._stop_session()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")
            finally:
                self._connection = None

        # Call parent close which handles event emission
        await super().close()
