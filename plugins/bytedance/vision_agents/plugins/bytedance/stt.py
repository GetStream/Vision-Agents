import asyncio
import logging
import time
from typing import Optional

import websockets
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

from . import _v3
from ._auth import DEFAULT_WS_HOST, Credentials

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
SAMPLE_RATE = 16000


class STT(stt.STT):
    """ByteDance / BytePlus Seed Speech streaming ASR.

    Uses the optimized bidirectional endpoint
    (``/api/v3/sauc/bigmodel_async``) of the Seed ASR 2.0 model. Server-side
    VAD segmentation (``enable_nonstream``) drives final transcripts and turn
    detection.

    Docs: https://docs.byteplus.com/en/docs/byteplusvoice/asrstreamingguide
    """

    turn_detection: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        access_key: Optional[str] = None,
        model_name: str = "bigmodel",
        enable_itn: bool = True,
        enable_punc: bool = True,
        enable_ddc: bool = False,
        ws_url: str = f"{DEFAULT_WS_HOST}/api/v3/sauc/bigmodel_async",
        resource_id: str = DEFAULT_RESOURCE_ID,
    ):
        """Initialize ByteDance STT.

        Args:
            api_key: New-console API key. Falls back to ``BYTEDANCE_API_KEY`` /
                ``BYTEPLUS_API_KEY``.
            app_key: Legacy-console app key. Falls back to ``BYTEDANCE_APP_KEY``.
            access_key: Legacy-console access key. Falls back to ``BYTEDANCE_ACCESS_KEY``.
            model_name: Seed ASR model cluster name. Defaults to ``bigmodel``.
            enable_itn: Inverse text normalization. Defaults to True.
            enable_punc: Punctuation. Defaults to True.
            enable_ddc: Semantic smoothing (disfluency removal). Defaults to False.
            ws_url: WebSocket endpoint. Override for a BytePlus-specific host.
            resource_id: ``X-Api-Resource-Id`` selecting the ASR SKU.
        """
        super().__init__(provider_name="bytedance")
        self._credentials = Credentials.resolve(api_key, app_key, access_key)
        self.model = model_name
        self._enable_itn = enable_itn
        self._enable_punc = enable_punc
        self._enable_ddc = enable_ddc
        self._ws_url = ws_url
        self._resource_id = resource_id

        self._ws: Optional[websockets.ClientConnection] = None
        self._connection_ready = asyncio.Event()
        self._listen_task: Optional[asyncio.Task] = None
        self._sequence = 1
        self._current_participant: Optional[Participant] = None
        self._audio_start_time: Optional[float] = None

    def _config_payload(self) -> dict:
        return {
            "user": {"uid": self.session_id},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": SAMPLE_RATE,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": self.model,
                "enable_itn": self._enable_itn,
                "enable_punc": self._enable_punc,
                "enable_ddc": self._enable_ddc,
                "enable_nonstream": True,
                "show_utterances": True,
                "result_type": "single",
            },
        }

    async def start(self):
        if self._ws is not None:
            logger.warning("ByteDance STT already started")
            return

        self._ws = await self._credentials.connect(
            self._ws_url, self._resource_id, "ByteDance STT"
        )

        await self._ws.send(_v3.build_full_client_request(self._config_payload()))
        self._on_connected()
        self._connection_ready.set()

        self._listen_task = asyncio.create_task(self._listen())
        await super().start()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        if self.closed:
            logger.warning("ByteDance STT is closed, ignoring audio")
            return

        await self._connection_ready.wait()
        if self._ws is None:
            return

        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        audio_bytes = pcm_data.resample(SAMPLE_RATE, 1).samples.tobytes()
        self._sequence += 1
        try:
            await self._ws.send(
                _v3.build_audio_only_request(audio_bytes, self._sequence)
            )
        except websockets.ConnectionClosed as e:
            self._emit_error_event(e, context="process_audio")

    async def _listen(self):
        ws = self._ws
        if ws is None:
            return
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    self._handle_message(_v3.parse_response(bytes(message)))
        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError as e:
            self._emit_error_event(e, context="listen")

    def _handle_message(self, message: _v3.Message) -> None:
        if message.type == _v3.MsgType.ERROR:
            error = RuntimeError(
                f"ByteDance ASR error {message.code}: {message.payload!r}"
            )
            self._emit_error_event(error, context="asr")
            return

        payload = message.payload
        if not isinstance(payload, dict):
            return

        result = payload.get("result")
        if not isinstance(result, dict):
            return

        participant = self._current_participant
        if participant is None:
            return

        utterances = result.get("utterances") or []
        definite = [u for u in utterances if u.get("definite")]
        response = TranscriptResponse(
            language=result.get("language"),
            model_name=self.model,
        )

        if definite:
            for utterance in definite:
                text = utterance.get("text", "")
                if not text:
                    continue
                self._emit_transcript_event(text, participant, response, mode="final")
                duration_ms = None
                if self._audio_start_time is not None:
                    duration_ms = (time.perf_counter() - self._audio_start_time) * 1000
                self._emit_turn_ended_event(participant, duration_ms=duration_ms)
                self._audio_start_time = None
            return

        text = result.get("text", "")
        if text:
            self._emit_transcript_event(text, participant, response, mode="replacement")

    async def close(self):
        if self._ws is not None:
            try:
                self._sequence += 1
                await self._ws.send(
                    _v3.build_audio_only_request(b"", self._sequence, last=True)
                )
            except websockets.ConnectionClosed:
                pass

        if self._listen_task is not None:
            await cancel_and_wait(self._listen_task)
            self._listen_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            self._on_disconnected()

        await super().close()
