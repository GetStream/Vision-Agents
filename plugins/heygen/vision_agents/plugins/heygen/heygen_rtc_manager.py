import asyncio
import logging
from typing import Optional, Callable, Any

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCConfiguration,
    MediaStreamTrack,
)

from .heygen_session import HeyGenSession
from .heygen_types import VideoQuality

logger = logging.getLogger(__name__)


class HeyGenRTCManager:
    """Manages WebRTC connection to HeyGen's Streaming Avatar API.

    Handles the low-level WebRTC peer connection, audio/video streaming,
    and communication with HeyGen's servers.
    """

    def __init__(
        self,
        avatar_id: str = "default",
        quality: Optional["VideoQuality"] = VideoQuality.HIGH,
        api_key: Optional[str] = None,
    ):
        """Initialize the RTC manager.

        Args:
            avatar_id: HeyGen avatar ID to use.
            quality: Video quality setting (VideoQuality.LOW, VideoQuality.MEDIUM, or VideoQuality.HIGH).
            api_key: HeyGen API key (uses HEYGEN_API_KEY env var if not provided).
        """
        # Default to HIGH if not provided
        if quality is None:
            quality = VideoQuality.HIGH

        self.session_manager = HeyGenSession(
            avatar_id=avatar_id,
            quality=quality,
            api_key=api_key,
        )

        self.pc: Optional[RTCPeerConnection] = None

        # Video track callback for receiving avatar video
        self._video_callback: Optional[Callable[[MediaStreamTrack], Any]] = None

        # Audio track callback for receiving avatar audio
        self._audio_callback: Optional[Callable[[MediaStreamTrack], Any]] = None

        self._connected = False
        self._connection_ready = asyncio.Event()

    async def connect(self) -> None:
        """Establish WebRTC connection to HeyGen's Streaming API.

        Sets up the peer connection, negotiates tracks, and establishes
        the connection for real-time avatar streaming.

        HeyGen flow:
        1. Create session -> HeyGen provides SDP offer and ICE servers
        2. Set HeyGen's offer as remote description
        3. Create answer
        4. Send answer to HeyGen
        5. Start session
        """
        try:
            # Create HeyGen session - they provide the SDP offer
            session_info = await self.session_manager.create_session()

            # Extract ICE servers and SDP offer from session info
            ice_servers = self._parse_ice_servers(session_info)

            # HeyGen's sdp field - check the actual structure
            sdp_data = session_info.get("sdp")

            if isinstance(sdp_data, dict):
                # Standard WebRTC format: {'type': 'offer', 'sdp': 'v=0...'}
                offer_sdp = sdp_data.get("sdp")
                sdp_type = sdp_data.get("type")
                logger.debug(f"Got SDP dict from HeyGen (type: {sdp_type})")
            elif isinstance(sdp_data, str) and sdp_data.startswith("v=0"):
                # Raw SDP string (less common)
                offer_sdp = sdp_data
                logger.debug("Got raw SDP string from HeyGen")
            else:
                offer_sdp = None

            if not offer_sdp:
                logger.error(f"Unexpected SDP format. Type: {type(sdp_data)}")
                if isinstance(sdp_data, dict):
                    logger.error(f"SDP dict keys: {list(sdp_data.keys())}")
                logger.error(f"SDP data: {str(sdp_data)[:200] if sdp_data else 'None'}")
                raise RuntimeError("No valid SDP offer received from HeyGen")

            # Create RTCPeerConnection with ICE servers
            config = RTCConfiguration(iceServers=ice_servers)
            self.pc = RTCPeerConnection(configuration=config)

            # Set up track handlers
            @self.pc.on("track")
            async def on_track(track: MediaStreamTrack):
                await self._handle_track(track)

            @self.pc.on("connectionstatechange")
            async def on_connection_state_change():
                if self.pc is None:
                    return
                logger.info(f"HeyGen connection state: {self.pc.connectionState}")
                if self.pc.connectionState == "connected":
                    self._connected = True
                    self._connection_ready.set()
                elif self.pc.connectionState in ["failed", "closed"]:
                    self._connected = False
                    self._connection_ready.clear()

            # Set HeyGen's offer as remote description
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
            await self.pc.setRemoteDescription(offer)

            # Log transceivers for debugging
            logger.debug(
                f"Transceivers after setRemoteDescription: {len(self.pc.getTransceivers())}"
            )

            # Create our answer
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)

            # Start the session with our SDP answer
            # HeyGen expects the answer in the start_session call
            await self.session_manager.start_session(
                sdp_answer=self.pc.localDescription.sdp
            )

            # Wait for connection to be established
            await asyncio.wait_for(self._connection_ready.wait(), timeout=10.0)

            logger.info("HeyGen WebRTC connection established")

        except Exception as e:
            logger.error(f"Failed to connect to HeyGen: {e}")
            raise

    def _parse_ice_servers(self, session_info: dict) -> list:
        """Parse ICE servers from HeyGen session info.

        HeyGen may provide ice_servers, ice_servers2, or rely on LiveKit's embedded servers.

        Args:
            session_info: Session information from HeyGen API.

        Returns:
            List of RTCIceServer objects.
        """
        ice_servers = []

        # Try ice_servers first, then ice_servers2 as backup
        ice_server_configs = (
            session_info.get("ice_servers")
            or session_info.get("ice_servers2")
            or session_info.get("iceServers", [])
        )

        if ice_server_configs and not isinstance(ice_server_configs, list):
            logger.warning(f"Unexpected ice_servers format: {type(ice_server_configs)}")
            ice_server_configs = []

        for server_config in ice_server_configs:
            if not isinstance(server_config, dict):
                continue

            urls = server_config.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]  # Convert single URL to list

            username = server_config.get("username")
            credential = server_config.get("credential")

            if urls:
                ice_servers.append(
                    RTCIceServer(
                        urls=urls,
                        username=username,
                        credential=credential,
                    )
                )
                logger.info(f"Added ICE server: {urls[0]}")

        # When using LiveKit, ICE servers may be embedded in SDP
        # In that case, use public STUN as fallback
        if not ice_servers:
            logger.info(
                "Using default STUN servers (LiveKit may provide its own via SDP)"
            )
            ice_servers.append(RTCIceServer(urls=["stun:stun.l.google.com:19302"]))

        return ice_servers

    async def _handle_track(self, track: MediaStreamTrack) -> None:
        """Handle incoming media track from HeyGen.

        Args:
            track: Incoming media track (audio or video).
        """
        logger.info(f"Received track from HeyGen: {track.kind}")

        if track.kind == "video":
            if self._video_callback:
                await self._video_callback(track)
            else:
                logger.warning("Video track received but no callback registered")
        elif track.kind == "audio":
            # Audio track from HeyGen (avatar speech with lip-synced TTS)
            logger.info("Audio track received from HeyGen")
            if self._audio_callback:
                await self._audio_callback(track)
            else:
                logger.warning("Audio track received but no callback registered")

    def set_video_callback(self, callback: Callable[[MediaStreamTrack], Any]) -> None:
        """Set callback for handling incoming video track.

        Args:
            callback: Async function to handle video track.
        """
        self._video_callback = callback

    def set_audio_callback(self, callback: Callable[[MediaStreamTrack], Any]) -> None:
        """Set callback for handling incoming audio track.

        Args:
            callback: Async function to handle audio track.
        """
        self._audio_callback = callback

    async def send_text(self, text: str, task_type: str = "repeat") -> None:
        """Send text to HeyGen for the avatar to speak with lip-sync.

        This is the correct way to achieve lip-sync with HeyGen - they handle
        TTS and lip-sync server-side based on the text input.

        Args:
            text: The text for the avatar to speak.
            task_type: Either "repeat" or "talk" (default: "repeat").
        """
        await self.session_manager.send_task(text, task_type)

    @property
    def is_connected(self) -> bool:
        """Check if WebRTC connection is established."""
        return self._connected

    async def close(self) -> None:
        """Close the WebRTC connection and clean up resources."""
        if self.pc:
            await self.pc.close()
            self.pc = None

        await self.session_manager.close()

        self._connected = False
        self._connection_ready.clear()

        logger.info("HeyGen RTC connection closed")
