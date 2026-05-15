"""WebRTC manager for Inworld Realtime API.

Handles low-level WebRTC peer connection, audio streaming, and data-channel
communication with Inworld's servers. Adapted from the OpenAI plugin's
RTCManager; Inworld's protocol is OpenAI-compatible but the signaling layer
(endpoint, auth, ICE fetch) differs.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import httpx
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from openai.types.realtime import RealtimeSessionCreateRequestParam
from vision_agents.core.utils.audio_forwarder import AudioForwarder

logger = logging.getLogger(__name__)

INWORLD_API_BASE = "https://api.inworld.ai"
INWORLD_CALLS_ENDPOINT = f"{INWORLD_API_BASE}/v1/realtime/calls"
INWORLD_ICE_ENDPOINT = f"{INWORLD_API_BASE}/v1/realtime/ice-servers"


class RTCManager:
    """Manages the WebRTC connection to Inworld's Realtime API.

    Handles the peer connection, audio streaming, and data-channel
    communication. Video is not supported (Inworld's docs do not document
    video input for realtime).
    """

    realtime_session: RealtimeSessionCreateRequestParam
    pc: RTCPeerConnection | None

    def __init__(
        self,
        api_key: str,
        realtime_session: RealtimeSessionCreateRequestParam,
    ):
        self._api_key = api_key
        self.realtime_session = realtime_session
        self.pc = None
        self.data_channel: RTCDataChannel | None = None
        self.call_id: str | None = None

        self._audio_to_inworld_track: AudioStreamTrack = AudioStreamTrack(
            sample_rate=48000
        )

        self._audio_callback: Callable[[PcmData], Awaitable[None]] | None = None
        self._event_callback: Callable[[dict], Awaitable[None]] | None = None
        self._data_channel_open_event: asyncio.Event = asyncio.Event()
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def connect(self) -> None:
        """Establish the WebRTC connection to Inworld.

        Pre-fetches TURN/STUN servers from Inworld so the peer connection can
        be created with them upfront — required because Inworld's media relay
        sits behind NAT (private IPs in ICE candidates) and cannot be reached
        without the server-issued TURN credentials.
        """
        ice_servers = await self._fetch_ice_servers()
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        try:
            self._setup_connection_logging()

            await self._add_data_channel()
            self.pc.addTrack(self._audio_to_inworld_track)

            @self.pc.on("track")
            async def on_track(track):
                if track.kind == "audio" and self._audio_callback:
                    audio_forwarder = AudioForwarder(track, self._audio_callback)
                    await audio_forwarder.start()

            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)

            answer_sdp = await self._exchange_sdp(offer.sdp)

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.pc.setRemoteDescription(answer)
        except BaseException:
            await self.close()
            raise

    async def send_audio_pcm(self, pcm: PcmData) -> None:
        """Send a PCM audio frame upstream."""
        await self._audio_to_inworld_track.write(pcm)

    async def send_text(self, text: str) -> None:
        """Send a text message and trigger a response."""
        await self.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self.send_event({"type": "response.create"})

    async def send_event(self, event: dict) -> None:
        """Send a JSON event through the Inworld data channel."""
        if not self.data_channel:
            logger.warning("Data channel not ready, cannot send event")
            return

        if not self._data_channel_open_event.is_set():
            try:
                await asyncio.wait_for(
                    self._data_channel_open_event.wait(), timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Data channel not open after timeout; dropping event")
                return

        if self.data_channel.readyState != "open":
            logger.warning(
                "Data channel state is %r, cannot send event",
                self.data_channel.readyState,
            )
            return

        self.data_channel.send(json.dumps(event))
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Sent event: %s", event.get("type"))

    async def _exchange_sdp(self, local_sdp: str) -> str:
        """POST the local SDP offer to Inworld and return the answer SDP.

        Raises:
            httpx.HTTPStatusError: non-2xx response from Inworld.
            httpx.RequestError: network/transport failure.
        """
        session_dict = dict(self.realtime_session)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                INWORLD_CALLS_ENDPOINT,
                json={"sdp": local_sdp, "session": session_dict},
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            resp.raise_for_status()
            body = resp.json()

        self.call_id = body.get("id")
        if self.call_id:
            logger.info("Inworld realtime call established (id=%s)", self.call_id)

        return body["sdp"]

    async def _fetch_ice_servers(self) -> list[RTCIceServer]:
        """Fetch ICE/TURN servers from Inworld.

        Inworld's media relay is behind NAT, so TURN is effectively required.
        Called before constructing the peer connection so the servers can be
        passed in via RTCConfiguration.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                INWORLD_ICE_ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            body = resp.json()

        raw = body.get("ice_servers") or body.get("iceServers") or []
        parsed = _parse_ice_servers(raw)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Inworld returned %d ICE server entries", len(parsed))
        return parsed

    async def _add_data_channel(self) -> None:
        assert self.pc is not None
        self.data_channel = self.pc.createDataChannel("oai-events", ordered=True)

        @self.data_channel.on("open")
        async def on_open():
            logger.info("Inworld data channel opened")
            self._data_channel_open_event.set()

        @self.data_channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.exception("Failed to decode data-channel message")
                return
            task = asyncio.create_task(self._handle_event(data))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def _handle_event(self, event: dict) -> None:
        cb = self._event_callback
        if cb is not None:
            await cb(event)

    def set_audio_callback(
        self, callback: Callable[[PcmData], Awaitable[None]]
    ) -> None:
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._event_callback = callback

    def _setup_connection_logging(self) -> None:
        assert self.pc is not None

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            assert self.pc is not None
            state = self.pc.connectionState
            if state == "failed":
                logger.error("Inworld RTC connection failed")
            elif state == "disconnected":
                logger.warning("Inworld RTC connection disconnected")
            elif state == "connected":
                logger.info("Inworld RTC connection established")
            elif state == "closed":
                logger.info("Inworld RTC connection closed")

        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            assert self.pc is not None
            state = self.pc.iceConnectionState
            if state == "failed":
                logger.error("Inworld ICE connection failed")
            elif state == "disconnected":
                logger.warning("Inworld ICE connection disconnected")
            elif state == "connected":
                logger.info("Inworld ICE connection established")

    async def close(self) -> None:
        if self.data_channel is not None:
            self.data_channel.close()
            self.data_channel = None
        self._audio_to_inworld_track.stop()

        pc = self.pc
        if pc is None:
            return
        self.pc = None

        try:
            await pc.close()
        except ConnectionError:
            logger.debug("Suppressed expected error closing Inworld peer connection")


def _parse_ice_servers(raw: list[dict]) -> list[RTCIceServer]:
    """Convert Inworld's ICE server dicts into aiortc RTCIceServer objects."""
    out: list[RTCIceServer] = []
    for entry in raw:
        urls = entry.get("urls") or entry.get("url")
        if not urls:
            continue
        out.append(
            RTCIceServer(
                urls=urls,
                username=entry.get("username"),
                credential=entry.get("credential"),
            )
        )
    return out
