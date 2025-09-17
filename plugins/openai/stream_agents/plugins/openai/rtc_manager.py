import asyncio
import json
from typing import Any, Optional, Callable
from os import getenv
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from httpx import AsyncClient, HTTPStatusError
import logging
from dotenv import load_dotenv
from getstream.video.rtc.track_util import PcmData

from aiortc.mediastreams import AudioStreamTrack
from fractions import Fraction
import numpy as np
from av import AudioFrame

load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI Realtime endpoints
OPENAI_REALTIME_BASE = "https://api.openai.com/v1/realtime"
OPENAI_SESSIONS_URL = f"{OPENAI_REALTIME_BASE}/sessions"


class RTCManager:
    def __init__(self, model: str, voice: str):
        self.api_key = getenv("OPENAI_API_KEY")
        self.model = model
        self.voice = voice
        self.token = None
        self.pc = RTCPeerConnection()
        self.data_channel: Optional[RTCDataChannel] = None
        self._mic_track: AudioStreamTrack = None
        self._audio_callback: Optional[Callable[[bytes], Any]] = None
        self._event_callback: Optional[Callable[[dict], Any]] = None
        self._data_channel_open_event: asyncio.Event = asyncio.Event()

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
        logger.info("Remote description set; WebRTC established")


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
            self._data_channel_open_event.set()

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

        @self.pc.on("track")
        async def on_track(track):
            if getattr(track, "kind", None) == "audio":
                logger.info("Remote audio track attached; starting reader")

                async def _reader():
                    while True:
                        try:
                            frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            logger.debug(f"Remote audio track ended or error: {e}")
                            break

                        try:
                            samples = frame.to_ndarray()
                            if samples.ndim == 2 and samples.shape[0] > 1:
                                samples = samples.mean(axis=0)
                            if samples.dtype != np.int16:
                                samples = (samples * 32767).astype(np.int16)
                            audio_bytes = samples.tobytes()
                            cb = self._audio_callback
                            if cb is not None:
                                if asyncio.iscoroutinefunction(cb):
                                    await cb(audio_bytes)
                                else:
                                    cb(audio_bytes)
                        except Exception as e:
                            logger.debug(f"Failed to process remote audio frame: {e}")

                asyncio.create_task(_reader())

    async def send_audio_pcm(self, pcm_data: PcmData) -> None:
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


    # async def send_text(self, text: str) -> None:
    #     """Send a text turn via the data channel and request a response."""
    #     if not self.data_channel:
    #         logger.warning("Data channel not ready; cannot send text")
    #         return
    #     try:
    #         evt_create = {
    #             "type": "conversation.item.create",
    #             "item": {
    #                 "type": "message",
    #                 "role": "user",
    #                 "content": [{"type": "input_text", "text": text}],
    #             },
    #         }
    #         self.data_channel.send(json.dumps(evt_create))
    #         print(f"Sent event: {evt_create}")
    #         self.data_channel.send(json.dumps({"type": "response.create"}))
    #         print("Requested response")
    #     except Exception as e:
    #         logger.error(f"Failed to send text over data channel: {e}")

    async def _setup_peer_connection_handlers(self) -> str:
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

    async def _handle_event(self, event: dict) -> None:
        """Minimal event handler for data channel messages."""
        logger.info(f"OpenAI event: {event}")
        cb = self._event_callback
        if cb is not None:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.debug(f"Event callback error: {e}")

    def set_audio_callback(self, callback: Callable[[bytes], Any]) -> None:
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable[[dict], Any]) -> None:
        self._event_callback = callback

    async def close(self) -> None:
        try:
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
                self._mic_track = None
            await self.pc.close()
        except Exception as e:
            logger.debug(f"RTCManager close error: {e}")
