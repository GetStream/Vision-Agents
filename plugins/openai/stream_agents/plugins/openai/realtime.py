import asyncio
import json
from typing import Any, Optional

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from httpx import AsyncClient, HTTPStatusError
from stream_agents.core.llm import realtime
import logging

from aiortc.mediastreams import AudioStreamTrack
from fractions import Fraction
import numpy as np
from av import AudioFrame

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
        self._mic_track: AudioStreamTrack = None

    async def connect(self) -> None:
        self.token = await self._get_session_token()
        logger.info("Obtained OpenAI session token")
        await self._add_data_channel()
        logger.info("Added data channel")
        await self._set_audio_track()
        logger.info("Set audio track for the call")
        answer_sdp = await self._setup_peer_connection_handlers()
        logger.info("Set up peer connection handlers")
        logger.info(f"Answer SDP: {answer_sdp}")
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

        @self.data_channel.on("open")
        def on_open():
            logger.info("Data channel opened")

        @self.data_channel.on("message")
        def on_message(message):
            try:
                data = json.loads(message)
                asyncio.create_task(self._handle_event(data))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")

    async def _set_audio_track(self) -> None:
        class MicAudioTrack(AudioStreamTrack):
            """Minimal audio track without a queue.

            - Generates 20 ms mono PCM16 silence by default
            - Accepts optional push-based PCM via set_input(bytes, sample_rate)
            - Drops old input if new input arrives (no buffering)
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
                        samples = arr[:1, :]
                    # Pad or truncate to exactly one 20 ms frame
                    needed = samples_per_frame
                    have = samples.shape[1]
                    if have < needed:
                        pad = np.zeros((1, needed - have), dtype=np.int16)
                        samples = np.concatenate([samples, pad], axis=1)
                    elif have > needed:
                        samples = samples[:, :needed]
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

        self._mic_track = MicAudioTrack(48000)
        self.pc.addTrack(self._mic_track)

    async def send_audio_pcm(self, pcm_data: bytes, sample_rate: int = 48000) -> None:
        if self._mic_track:
            try:
                self._mic_track.set_input(pcm_data, sample_rate)
            except Exception as e:
                logger.error(f"Failed to push mic audio: {e}")

    async def _setup_peer_connection_handlers(self) -> str:
        # Set remote track handler
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

    async def send_audio_pcm(self, audio: bytes, sample_rate: int = 48000):
        await self.rtc.send_audio_pcm(audio, sample_rate)

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