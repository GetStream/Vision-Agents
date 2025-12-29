import asyncio
import json
from typing import Any, Optional, Callable, cast, Literal

import av
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
    RTCRtpSender,
)
from openai import AsyncOpenAI
from openai.types.realtime import RealtimeSessionCreateRequestParam
from openai.types.beta.realtime import (
    ConversationItemCreateEvent,
    ConversationItem,
    ConversationItemContent,
)

from getstream.video.rtc.audio_track import AudioStreamTrack
import logging
from getstream.video.rtc.track_util import PcmData

from aiortc.mediastreams import MediaStreamTrack

from vision_agents.core.utils.audio_forwarder import AudioForwarder
from vision_agents.core.utils.audio_track import QueuedAudioTrack
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)


class RTCManager:
    """Manages WebRTC connection to OpenAI's Realtime API.

    Handles the low-level WebRTC peer connection, audio/video streaming,
    and data channel communication with OpenAI's servers.
    """

    realtime_session: RealtimeSessionCreateRequestParam
    client: AsyncOpenAI
    pc: RTCPeerConnection

    def __init__(
        self,
        realtime_session: RealtimeSessionCreateRequestParam,
        client: AsyncOpenAI,
        send_video: bool,
    ):
        self.realtime_session = realtime_session
        self.client = client
        self.send_video = send_video
        self.pc = RTCPeerConnection()
        self.data_channel: Optional[RTCDataChannel] = None

        # tracks for sharing audio & video
        self._audio_to_openai_track: QueuedAudioTrack = QueuedAudioTrack(
            sample_rate=48000
        )
        self._video_to_openai_track: QueuedVideoTrack = QueuedVideoTrack()
        self._video_sender: Optional[RTCRtpSender] = None

        # Set up connection event handlers
        self._setup_connection_logging()
        self._audio_callback: Optional[Callable[[PcmData], Any]] = None
        self._event_callback: Optional[Callable[[dict], Any]] = None
        self._data_channel_open_event: asyncio.Event = asyncio.Event()
        self._current_video_forwarder = None

    async def connect(self) -> None:
        """Establish WebRTC connection to OpenAI's Realtime API.

        Sets up the peer connection, negotiates audio and video tracks,
        and establishes the data channel for real-time communication.
        """
        await self._add_data_channel()
        await self._set_audio_track()

        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                track = cast(AudioStreamTrack, track)
                if self._audio_callback:
                    audio_forwarder = AudioForwarder(track, self._audio_callback)
                    await audio_forwarder.start()

        # TODO: this is not ideal.. but we can't renegotiate since you lose the session/memory
        # see https://platform.openai.com/docs/api-reference/realtime/create-call
        # setting video sender skips the renegotiate
        if self.send_video:
            self._video_sender = self.pc.addTrack(self._video_to_openai_track)
        await self.renegotiate()

    async def send_audio_pcm(self, pcm: PcmData) -> None:
        await self._audio_to_openai_track.write(pcm)

    async def send_text(
        self, text: str, role: Literal["user", "assistant", "system"] = "user"
    ):
        event_type = ConversationItemCreateEvent(
            type="conversation.item.create",
            item=ConversationItem(
                type="message",
                role=role,
                content=[
                    ConversationItemContent(
                        type="input_text",
                        text=text,
                    )
                ],
            ),
        )
        event = event_type.to_dict()
        await self._send_event(event)
        # Explicitly request audio response for this turn using top-level fields
        await self._send_event(
            {
                "type": "response.create",
            }
        )

    async def start_video_sender(
        self, stream_video_track: MediaStreamTrack, fps: int = 1, shared_forwarder=None
    ) -> None:
        await self._set_video_track()

        # This method can be called twice with different forwarders
        # Remove handler from old forwarder if it exists
        if self._current_video_forwarder is not None:
            await self._current_video_forwarder.remove_frame_handler(
                self._send_video_frame
            )
            logger.debug("Removed old video frame handler from previous forwarder")

        # Create a VideoForwarder if one wasn't provided
        if shared_forwarder is None:
            shared_forwarder = VideoForwarder(
                input_track=stream_video_track,  # type: ignore[arg-type]
                max_buffer=10,
                fps=float(fps),
                name="openai_rtc_forwarder",
            )
            logger.info("Created new VideoForwarder for OpenAI RTC")

        # Store reference to new forwarder and add handler
        self._current_video_forwarder = shared_forwarder
        shared_forwarder.add_frame_handler(
            self._send_video_frame, fps=float(fps), name="openai"
        )

    def _setup_connection_logging(self) -> None:
        """Set up event handlers for connection monitoring and error logging."""

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = self.pc.connectionState
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

        @self.pc.on("signalingstatechange")
        async def on_signalingstatechange():
            state = self.pc.signalingState
            logger.debug(f"📡 Signaling state: {state}")

        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            logger.debug(f"📨 Remote data channel created: {channel.label}")

    async def renegotiate(self) -> None:
        """Renegotiate the connection with OpenAI.

        Public method to repeat the offer/answer cycle, useful when tracks are added
        or modified after the initial connection.
        """
        # Create local offer and exchange SDP
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        answer_sdp = await self._exchange_sdp(offer.sdp)
        if not answer_sdp:
            raise RuntimeError("Failed to get remote SDP from OpenAI")

        # Set the remote SDP we got from OpenAI
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)

    async def _add_data_channel(self) -> None:
        # Add data channel
        self.data_channel = self.pc.createDataChannel("oai-events")

        @self.data_channel.on("open")
        async def on_open():
            self._data_channel_open_event.set()

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
            if self._video_sender is None:
                logger.debug("_set_video_track enableing addTrack")
                self._video_sender = self.pc.addTrack(self._video_to_openai_track)
                # adding tracks requires renegotiation
                await self.renegotiate()

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

    async def _exchange_sdp(self, local_sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI using the realtime calls API."""
        logger.debug(f"Creating realtime call with SDP length: {len(local_sdp)} bytes")

        # Use the OpenAI client's realtime calls API
        response = await self.client.realtime.calls.create(
            session=self.realtime_session, sdp=local_sdp
        )

        logger.debug("SDP response from OpenAI")
        return response.text

    async def _handle_event(self, event: dict) -> None:
        cb = self._event_callback
        if cb is not None:
            await cb(event)

    def set_audio_callback(self, callback: Callable[[PcmData], Any]) -> None:
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable[[dict], Any]) -> None:
        self._event_callback = callback

    async def close(self) -> None:
        if self.data_channel is not None:
            self.data_channel.close()
            self.data_channel = None
        self._audio_to_openai_track.stop()
        if self._video_to_openai_track is not None:
            self._video_to_openai_track.stop()

        async def _safe_close():
            try:
                await self.pc.close()
            except Exception as e:
                logger.error(f"Error closing peer connection: {e}")

        asyncio.create_task(_safe_close())
