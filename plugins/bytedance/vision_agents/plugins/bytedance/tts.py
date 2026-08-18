import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Optional

import websockets
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import tts

from . import _v3
from ._auth import DEFAULT_WS_HOST, Credentials

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"
NAMESPACE = "BidirectionalTTS"


class TTS(tts.TTS):
    """ByteDance / BytePlus Seed Speech bidirectional streaming TTS.

    Keeps a single WebSocket connection open and opens one session per
    ``stream_audio`` call. PCM audio is requested directly so no opus/mp3
    decoding is needed.

    Docs: https://www.volcengine.com/docs/6561/1329505
    """

    streaming: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        access_key: Optional[str] = None,
        speaker: str = DEFAULT_SPEAKER,
        sample_rate: int = 24000,
        speech_rate: int = 0,
        ws_url: str = f"{DEFAULT_WS_HOST}/api/v3/tts/bidirection",
        resource_id: str = DEFAULT_RESOURCE_ID,
    ):
        """Initialize ByteDance TTS.

        Args:
            api_key: New-console API key. Falls back to ``BYTEDANCE_API_KEY`` /
                ``BYTEPLUS_API_KEY``.
            app_key: Legacy-console app key. Falls back to ``BYTEDANCE_APP_KEY``.
            access_key: Legacy-console access key. Falls back to ``BYTEDANCE_ACCESS_KEY``.
            speaker: Voice identifier. Defaults to a public bigtts voice.
            sample_rate: Output PCM sample rate in Hz. Defaults to 24000.
            speech_rate: Speaking-rate adjustment in [-50, 100]. Defaults to 0.
            ws_url: WebSocket endpoint. Override for a BytePlus-specific host.
            resource_id: ``X-Api-Resource-Id`` selecting the TTS SKU.
        """
        super().__init__(provider_name="bytedance")
        self._credentials = Credentials.resolve(api_key, app_key, access_key)
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.speech_rate = speech_rate
        self._ws_url = ws_url
        self._resource_id = resource_id

        self._ws: Optional[websockets.ClientConnection] = None
        self._connect_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._generation = 0

    def _req_params(self, text: Optional[str] = None) -> dict:
        audio_params = {"format": "pcm", "sample_rate": self.sample_rate}
        if self.speech_rate:
            audio_params["speech_rate"] = self.speech_rate
        req_params: dict = {"speaker": self.speaker, "audio_params": audio_params}
        if text is not None:
            req_params["text"] = text
        return req_params

    async def _ensure_connection(self) -> websockets.ClientConnection:
        if self._ws is not None:
            return self._ws

        async with self._connect_lock:
            if self._ws is not None:
                return self._ws

            headers = self._credentials.headers(self._resource_id)
            ws = await websockets.connect(
                self._ws_url,
                additional_headers=headers,
                max_size=10 * 1024 * 1024,
            )
            await ws.send(
                _v3.build_event_message(
                    _v3.MsgType.FULL_CLIENT_REQUEST,
                    _v3.EventType.START_CONNECTION,
                    payload=b"{}",
                )
            )
            raw = await ws.recv()
            raw_bytes = raw if isinstance(raw, bytes) else raw.encode()
            started = _v3.parse_response(raw_bytes)
            if started.event != _v3.EventType.CONNECTION_STARTED:
                await ws.close()
                raise RuntimeError(
                    f"ByteDance TTS handshake failed: {started.event} {started.payload}"
                )
            self._ws = ws
            self._on_connected()
            logger.debug("ByteDance TTS connected at %dHz", self.sample_rate)
            return ws

    async def stream_audio(self, text: str, *args, **kwargs) -> AsyncIterator[PcmData]:
        ws = await self._ensure_connection()
        self._stop_event.clear()
        self._generation += 1
        generation = self._generation

        session_id = str(uuid.uuid4())
        start_payload = json.dumps(
            {"namespace": NAMESPACE, "req_params": self._req_params()}
        ).encode()
        task_payload = json.dumps(
            {"namespace": NAMESPACE, "req_params": self._req_params(text)}
        ).encode()

        await ws.send(
            _v3.build_event_message(
                _v3.MsgType.FULL_CLIENT_REQUEST,
                _v3.EventType.START_SESSION,
                payload=start_payload,
                session_id=session_id,
            )
        )
        await ws.send(
            _v3.build_event_message(
                _v3.MsgType.FULL_CLIENT_REQUEST,
                _v3.EventType.TASK_REQUEST,
                payload=task_payload,
                session_id=session_id,
            )
        )
        await ws.send(
            _v3.build_event_message(
                _v3.MsgType.FULL_CLIENT_REQUEST,
                _v3.EventType.FINISH_SESSION,
                payload=b"{}",
                session_id=session_id,
            )
        )
        return self._receive_audio(session_id, generation)

    async def _receive_audio(
        self, session_id: str, generation: int
    ) -> AsyncIterator[PcmData]:
        ws = self._ws
        if ws is None:
            return

        async for message in ws:
            if self._stop_event.is_set() or generation != self._generation:
                return
            if not isinstance(message, (bytes, bytearray)):
                continue

            parsed = _v3.parse_response(bytes(message))
            if parsed.session_id is not None and parsed.session_id != session_id:
                continue

            if parsed.type == _v3.MsgType.ERROR:
                raise RuntimeError(
                    f"ByteDance TTS error {parsed.code}: {parsed.payload}"
                )
            if parsed.event == _v3.EventType.TTS_RESPONSE and isinstance(
                parsed.payload, (bytes, bytearray)
            ):
                if parsed.payload:
                    yield PcmData.from_bytes(
                        bytes(parsed.payload), sample_rate=self.sample_rate, channels=1
                    )
            elif parsed.event in (
                _v3.EventType.SESSION_FINISHED,
                _v3.EventType.SESSION_CANCELED,
            ):
                return
            elif parsed.event == _v3.EventType.SESSION_FAILED:
                raise RuntimeError(f"ByteDance TTS session failed: {parsed.payload}")

    async def stop_audio(self) -> None:
        self._stop_event.set()
        self._generation += 1

    async def close(self) -> None:
        await super().close()
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            self._on_disconnected()
