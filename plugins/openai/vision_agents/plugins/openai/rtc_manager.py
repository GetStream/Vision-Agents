import asyncio
import json
from typing import Any, Optional, Callable, cast
from os import getenv

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from getstream.video.rtc.audio_track import AudioStreamTrack
from httpx import AsyncClient, HTTPStatusError
import logging
from getstream.video.rtc.track_util import PcmData

from aiortc.mediastreams import MediaStreamTrack

from vision_agents.core.utils.audio_forwarder import AudioForwarder
from vision_agents.core.utils.audio_track import QueuedAudioTrack
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)

# OpenAI Realtime endpoints
OPENAI_REALTIME_BASE = "https://api.openai.com/v1/realtime"
OPENAI_SESSIONS_URL = f"{OPENAI_REALTIME_BASE}/sessions"


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

        # Set up connection event handlers
        self._setup_connection_logging()

        # on this track we send audio to openAI
        self._audio_to_openai_track: QueuedAudioTrack = QueuedAudioTrack(
            sample_rate=48000
        )
        # video to openAI
        self._video_to_openai_track: Optional[QueuedVideoTrack] = QueuedVideoTrack(
            width=640, height=480
        )

        self._audio_callback: Optional[Callable[[PcmData], Any]] = None
        self._event_callback: Optional[Callable[[dict], Any]] = None
        self._data_channel_open_event: asyncio.Event = asyncio.Event()
        self.send_video = send_video

        self.instructions: Optional[str] = None
        self._current_video_forwarder = None

    def _setup_connection_logging(self) -> None:
        """Set up event handlers for connection monitoring and error logging."""

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self.pc.connectionState
            logger.info(f"🔗 RTC connection state changed: {state}")
            if state == "failed":
                logger.error("❌ RTC connection failed")
            elif state == "disconnected":
                logger.warning("⚠️ RTC connection disconnected")
            elif state == "connected":
                logger.info("✅ RTC connection established")
            elif state == "closed":
                logger.info("🔒 RTC connection closed")

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            state = self.pc.iceConnectionState
            logger.info(f"🧊 ICE connection state: {state}")
            if state == "failed":
                logger.error("❌ ICE connection failed")
            elif state == "disconnected":
                logger.warning("⚠️ ICE connection disconnected")
            elif state == "connected":
                logger.info("✅ ICE connection established")
            elif state == "checking":
                logger.debug("🔍 ICE checking candidates...")

        @self.pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            state = self.pc.iceGatheringState
            logger.debug(f"🧊 ICE gathering state: {state}")
            if state == "complete":
                logger.info("✅ ICE gathering complete")

        @self.pc.on("signalingstatechange")
        async def on_signalingstatechange():
            state = self.pc.signalingState
            logger.debug(f"📡 Signaling state: {state}")

        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            logger.info(f"📨 Remote data channel created: {channel.label}")

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
            logger.info("receiving track from openai")
            if track.kind == "audio":
                track = cast(AudioStreamTrack, track)
                if self._audio_callback:
                    audio_forwarder = AudioForwarder(track, self._audio_callback)
                    await audio_forwarder.start()

        answer_sdp = await self._setup_sdp_exchange()

        # Set the remote SDP we got from OpenAI TODO: shouldnt we repeat this after track changes?
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
                    resp = await client.post(
                        url, headers=headers, json=payload, timeout=15
                    )
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

            # Immediately switch to semantic VAD and enable input audio transcription
            await self._send_event(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": {"type": "semantic_vad"},
                        "input_audio_transcription": {"model": "whisper-1"},
                    },
                }
            )
            # Session information will be automatically stored when session.created event is received
            logger.info(
                "Requested semantic_vad and input_audio_transcription via session.update"
            )

        @self.data_channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                asyncio.create_task(self._handle_event(data))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")

    async def _set_audio_track(self) -> None:
        self.pc.addTrack(self._audio_to_openai_track)

    async def _set_video_track(self) -> None:
        if self._video_to_openai_track:
            self._video_sender = self.pc.addTrack(self._video_to_openai_track)

    async def send_audio_pcm(self, pcm: PcmData) -> None:
        await self._audio_to_openai_track.write(pcm)

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
                    await asyncio.wait_for(
                        self._data_channel_open_event.wait(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Data channel not open after timeout; dropping event"
                    )
                    return

            if self.data_channel.readyState and self.data_channel.readyState != "open":
                logger.warning(
                    f"Data channel state is '{self.data_channel.readyState}', cannot send event"
                )

            message_json = json.dumps(event)
            self.data_channel.send(message_json)
            logger.debug(f"Sent event: {event.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send event: {e}")

    async def _send_video_frame(self, frame: av.VideoFrame) -> None:
        """
        Send a video frame to Gemini using send_realtime_input
        """
        logger.debug(f"Sending video frame: {frame}")
        if self._video_to_openai_track:
            await self._video_to_openai_track.add_frame(frame)

    async def start_video_sender(
        self, stream_video_track: MediaStreamTrack, fps: int = 1, shared_forwarder=None
    ) -> None:
        """Replace dummy video track with the actual Stream Video forwarding track.

        This creates a forwarding track that reads frames from the Stream Video track
        and forwards them through the OpenAI WebRTC connection.

        Args:
            stream_video_track: Video track to forward to OpenAI.
            fps: Target frames per second.
            shared_forwarder: Optional shared VideoForwarder to use instead of creating a new one.
        """
        if not self.send_video:
            logger.error("❌ Video sending not enabled for this session")
            raise RuntimeError("Video sending not enabled for this session")

        # This method can be called twice with different forwarders
        # Remove handler from old forwarder if it exists
        if self._current_video_forwarder is not None:
            await self._current_video_forwarder.remove_frame_handler(self._send_video_frame)
            logger.debug("Removed old video frame handler from previous forwarder")

        # Store reference to new forwarder and add handler
        self._current_video_forwarder = shared_forwarder
        shared_forwarder.add_frame_handler(
            self._send_video_frame, fps=float(fps), name="openai"
        )

        logger.info(
            f"✅ Successfully replaced OpenAI track with Stream Video forwarding (fps={fps})"
        )

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
                response = await client.post(
                    url, headers=headers, content=local_sdp, timeout=20
                )
                response.raise_for_status()
                return response.text if response.text else None
        except HTTPStatusError as e:
            body = e.response.text if e.response is not None else ""
            logger.error(f"SDP exchange failed: {e}; body={body}")
            raise
        except Exception as e:
            logger.error(f"SDP exchange failed: {e}")
            raise

    async def _handle_event(self, event: dict) -> None:
        """Minimal event handler for data channel messages."""
        cb = self._event_callback
        if cb is not None:
            await cb(event)

        # Store session information when we receive session.created event
        # FIXME Typing
        if event.get("type") == "session.created" and "session" in event:
            self.session_info = event["session"]
            logger.info(f"Stored session info: {self.session_info}")

    async def request_session_info(self) -> None:
        """Request and log current session information.

        Note:
            Session information is automatically stored when session.created event is received.
            This method only logs the stored information.
        """
        if self.session_info:
            logger.info(f"Current session info: {self.session_info}")
        else:
            logger.info(
                "No session information available yet. Waiting for session.created event."
            )

    def set_audio_callback(self, callback: Callable[[PcmData], Any]) -> None:
        """Set callback for receiving audio data from OpenAI.

        Args:
            callback: Function that receives PcmData (16kHz, mono, int16) from OpenAI responses.
        """
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable[[dict], Any]) -> None:
        """Set callback for receiving events from OpenAI.

        Args:
            callback: Function that receives event dicts from the OpenAI data channel.
        """
        self._event_callback = callback

    async def close(self) -> None:
        """Close the WebRTC connection and clean up resources."""
        try:
            if self.data_channel is not None:
                self.data_channel.close()
                self.data_channel = None
            if self._audio_to_openai_track is not None:
                self._audio_to_openai_track.stop()
            if self._video_to_openai_track is not None:
                self._video_to_openai_track.stop()
            await self.pc.close()
        except Exception as e:
            logger.debug(f"RTCManager close error: {e}")
