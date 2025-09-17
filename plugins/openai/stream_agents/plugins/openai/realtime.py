import asyncio
import json
from typing import Any, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from httpx import AsyncClient, HTTPStatusError
from stream_agents.core.llm import realtime
import logging

logger = logging.getLogger(__name__)

# OpenAI Realtime endpoints
OPENAI_REALTIME_BASE = "https://api.openai.com/v1/realtime"
OPENAI_SESSIONS_URL = f"{OPENAI_REALTIME_BASE}/sessions"


class RTCManager:
    def __init__(self, api_key: str, model: str, voice: str):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.token = None
        self.pc = RTCPeerConnection()
        self.data_channel: Optional[RTCDataChannel] = None

    async def connect(self) -> None:
        self.token = await self._get_session_token()
        logger.info("Obtained OpenAI session token")
        await self._add_data_channel()
        logger.info("Added data channel")
        answer_sdp = await self._setup_peer_connection_handlers()
        logger.info("Set up peer connection handlers")
        # WE ARE HERE
        # Set the remote SDP we got from OpenAI
        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self.pc.setRemoteDescription(answer)

              
    async def _get_session_token(self) -> str | None:
        url = OPENAI_SESSIONS_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "voice": self.voice}

        async with AsyncClient() as client:
            for attempt in range(2):
                try:
                    resp = await client.post(url, headers=headers, json=payload, timeout=15)
                    resp.raise_for_status()
                    data: dict = resp.json()
                    secret = data.get("client_secret")
                    return secret.get("value")
                except Exception as e:
                    if attempt == 0:
                        await asyncio.sleep(1.0)
                        continue
                    logger.error(f"Failed to get OpenAI Realtime session token: {e}")
                    return None
            return None

    async def _add_data_channel(self) -> None:
        # Add data channel
        self.data_channel = self.pc.createDataChannel("oai-events")

        @self.data_channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                asyncio.create_task(self._handle_event(data))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")

    async def _setup_peer_connection_handlers(self) -> str:
        # Set remote track handler first
        @self.pc.on("track")
        async def on_track(track):
            if track.kind == "audio":
                logger.debug("Remote audio track attached")

        # Create local offer and exchange SDP
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        answer_sdp = await self._exchange_sdp(offer.sdp)
        if not answer_sdp:
            raise RuntimeError("Failed to get remote SDP from OpenAI")
        return answer_sdp

    async def _exchange_sdp(self, local_sdp: str) -> Optional[str]:
        """Exchange SDP with OpenAI."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
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

    async def _handle_event(self, event: dict) -> None:
        """Minimal event handler for data channel messages."""
        logger.info(f"OpenAI event: {event}")


class Realtime(realtime.Realtime):
    def __init__(self, api_key: str, model: str, voice: str):
        super().__init__()
        self.rtc = RTCManager(api_key, model, voice)

    async def connect(self):
        await self.rtc.connect()

    # def send_audio_pcm(self, audio):
    #     pass

    async def send_text(self, text):
        pass

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        ...

    async def _close_impl(self):
        ...