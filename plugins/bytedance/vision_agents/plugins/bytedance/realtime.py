import asyncio
import logging
from enum import IntEnum
from typing import AsyncIterator, Optional

import websockets
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import AudioInputPacingConfig
from vision_agents.core.llm import Realtime as CoreRealtime
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal

from . import _ast
from ._auth import DEFAULT_WS_HOST, Credentials

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_ID = "volc.service_type.10053"
INPUT_SAMPLE_RATE = 16000


class AstEvent(IntEnum):
    START_SESSION = 100
    FINISH_SESSION = 102
    SESSION_STARTED = 150
    SESSION_CANCELED = 151
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    TTS_SENTENCE_START = 350
    TTS_SENTENCE_END = 351
    TTS_RESPONSE = 352
    SOURCE_SUBTITLE_START = 650
    SOURCE_SUBTITLE_RESPONSE = 651
    SOURCE_SUBTITLE_END = 652
    TRANSLATION_SUBTITLE_START = 653
    TRANSLATION_SUBTITLE_RESPONSE = 654
    TRANSLATION_SUBTITLE_END = 655


class Realtime(CoreRealtime):
    """ByteDance / BytePlus Seed Speech Live Interpretation (AST 2.0).

    A speech-to-speech translator: audio in the source language streams in,
    translated audio plus source/translation subtitles stream out. Use it in
    the same Agent slot as other realtime models (``llm=bytedance.Realtime(...)``),
    not as a chat LLM — text prompts and function calling are not supported.

    Docs: https://docs.byteplus.com/en/docs/byteplusvoice/liveinterpretationapi
    """

    provider_name = "bytedance"

    def __init__(
        self,
        source_language: str,
        target_language: str,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        access_key: Optional[str] = None,
        speaker: Optional[str] = None,
        mode: str = "s2s",
        sample_rate: int = 24000,
        ws_url: str = f"{DEFAULT_WS_HOST}/api/v4/ast/v2/translate",
        resource_id: str = DEFAULT_RESOURCE_ID,
        input_audio_pacing: Optional[AudioInputPacingConfig] = None,
    ):
        """Initialize ByteDance Live Interpretation.

        Args:
            source_language: Source language code (e.g. ``zh``, ``en``).
            target_language: Target language code. At least one of source/target
                must be ``zh`` or ``en``.
            api_key: New-console API key. Falls back to ``BYTEDANCE_API_KEY`` /
                ``BYTEPLUS_API_KEY``.
            app_key: Legacy-console app key. Falls back to ``BYTEDANCE_APP_KEY``.
            access_key: Legacy-console access key. Falls back to ``BYTEDANCE_ACCESS_KEY``.
            speaker: Optional voice id; empty keeps the source speaker's timbre.
            mode: ``s2s`` for translated speech + subtitles, ``s2t`` for subtitles only.
            sample_rate: Output PCM sample rate in Hz. Defaults to 24000.
            ws_url: WebSocket endpoint. Override for a BytePlus-specific host.
            resource_id: ``X-Api-Resource-Id`` selecting the AST SKU.
            input_audio_pacing: Input pacing config. Defaults to a virtual
                microphone (steady cadence with silence fill) which AST needs.
        """
        if mode not in ("s2s", "s2t"):
            raise ValueError("mode must be 's2s' or 's2t'")
        if "zh" not in (source_language, target_language) and "en" not in (
            source_language,
            target_language,
        ):
            raise ValueError(
                "At least one of source_language / target_language must be 'zh' or 'en'"
            )

        super().__init__(
            input_audio_pacing=input_audio_pacing
            or AudioInputPacingConfig.virtual_microphone()
        )
        self._credentials = Credentials.resolve(api_key, app_key, access_key)
        self.source_language = source_language
        self.target_language = target_language
        self.speaker = speaker
        self.mode = mode
        self.sample_rate = sample_rate
        self._ws_url = ws_url
        self._resource_id = resource_id

        self._ws: Optional[websockets.ClientConnection] = None
        self._processing_task: Optional[asyncio.Task] = None
        self._session_started = asyncio.Event()

    def _start_session_request(self) -> bytes:
        target_audio = None
        if self.mode == "s2s":
            target_audio = _ast.Audio(
                format="pcm", rate=self.sample_rate, bits=16, channel=1
            )
        return _ast.TranslateRequest(
            event=AstEvent.START_SESSION,
            session_id=self.session_id,
            user_uid="vision-agents",
            source_audio=_ast.Audio(
                format="pcm", codec="raw", rate=INPUT_SAMPLE_RATE, bits=16, channel=1
            ),
            target_audio=target_audio,
            request=_ast.ReqParams(
                mode=self.mode,
                source_language=self.source_language,
                target_language=self.target_language,
                speaker_id=self.speaker or "",
            ),
            denoise=False,
        ).encode()

    async def connect(self):
        if self._ws is not None:
            return

        headers = self._credentials.headers(self._resource_id)
        self._ws = await websockets.connect(
            self._ws_url,
            additional_headers=headers,
            max_size=10 * 1024 * 1024,
        )
        response = self._ws.response
        logid = response.headers.get("X-Tt-Logid") if response is not None else None
        logger.debug("ByteDance Realtime connected, logid=%s", logid)

        await self._ws.send(self._start_session_request())
        self._on_connected(
            session_config={
                "mode": self.mode,
                "source_language": self.source_language,
                "target_language": self.target_language,
            }
        )
        self._processing_task = asyncio.create_task(self._process_events())

    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ):
        if not self.connected or self._ws is None:
            return
        if not self._session_started.is_set():
            return

        self._current_participant = participant
        audio_bytes = pcm.resample(INPUT_SAMPLE_RATE, 1).samples.tobytes()
        req = _ast.TranslateRequest(
            event=AstEvent.TASK_REQUEST,
            session_id=self.session_id,
            source_audio=_ast.Audio(binary_data=audio_bytes),
        )
        try:
            await self._ws.send(req.encode())
        except websockets.ConnectionClosed as e:
            self._emit_error_event(e, context="simple_audio_response")

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        logger.warning(
            "bytedance.Realtime is a live interpreter and does not accept text prompts"
        )
        yield LLMResponseFinal()

    async def watch_video_track(self, track, shared_forwarder=None) -> None:
        logger.warning("bytedance.Realtime does not support video input")

    async def _process_events(self):
        ws = self._ws
        if ws is None:
            return
        try:
            async for message in ws:
                if isinstance(message, (bytes, bytearray)):
                    self._handle_response(bytes(message))
        except websockets.ConnectionClosedOK:
            pass
        except websockets.ConnectionClosedError as e:
            self._emit_error_event(e, context="process_events")

    def _handle_response(self, data: bytes) -> None:
        response = _ast.TranslateResponse.decode(data)
        event = response.event

        if event == AstEvent.SESSION_STARTED:
            self._session_started.set()
            logger.debug("ByteDance Realtime session started")
        elif event == AstEvent.SESSION_FAILED:
            self._emit_error_event(
                RuntimeError(f"AST session failed: {response.message}"),
                context="ast",
            )
        elif event in (AstEvent.SESSION_FINISHED, AstEvent.SESSION_CANCELED):
            logger.debug("ByteDance Realtime session finished")
        elif event == AstEvent.SOURCE_SUBTITLE_RESPONSE:
            if response.text:
                self._emit_user_speech_transcription(response.text, mode="replacement")
        elif event == AstEvent.SOURCE_SUBTITLE_END:
            if response.text:
                self._emit_user_speech_transcription(response.text, mode="final")
        elif event == AstEvent.TRANSLATION_SUBTITLE_RESPONSE:
            if response.text:
                self._emit_agent_speech_transcription(response.text, mode="replacement")
        elif event == AstEvent.TRANSLATION_SUBTITLE_END:
            if response.text:
                self._emit_agent_speech_transcription(response.text, mode="final")
        elif event == AstEvent.TTS_SENTENCE_START:
            self._emit_agent_speech_started()
        elif event == AstEvent.TTS_SENTENCE_END:
            self._emit_agent_speech_ended()
        elif event == AstEvent.TTS_RESPONSE:
            if response.data:
                pcm = PcmData.from_bytes(
                    response.data, sample_rate=self.sample_rate, channels=1
                )
                self._emit_audio_output_event(pcm=pcm)

    async def close(self):
        if self._ws is not None:
            try:
                req = _ast.TranslateRequest(
                    event=AstEvent.FINISH_SESSION, session_id=self.session_id
                )
                await self._ws.send(req.encode())
            except websockets.ConnectionClosed:
                pass

        if self._processing_task is not None:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        await self._close_audio_input()

        if self._ws is not None:
            await self._ws.close()
            self._ws = None

        self._on_disconnected()
