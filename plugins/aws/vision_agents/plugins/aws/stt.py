import asyncio
import logging
import time
from typing import Any, Optional

from aws_sdk_transcribe_streaming.client import (
    Config,
    StartStreamTranscriptionInput,
    StartStreamTranscriptionOutput,
    TranscribeStreamingClient,
)
from aws_sdk_transcribe_streaming.models import (
    AudioEvent,
    AudioStream,
    AudioStreamAudioEvent,
    Result,
    TranscriptEvent,
    TranscriptResultStream,
    TranscriptResultStreamTranscriptEvent,
)
from getstream.video.rtc.track_util import PcmData
from smithy_core.aio.eventstream import DuplexEventStream
from smithy_core.aio.interfaces.eventstream import EventPublisher, EventReceiver
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

from ._credentials import Boto3CredentialsResolver

# TODO(stt): reconnection on transport errors is deferred (see plan).

logger = logging.getLogger(__name__)


class TranscribeSTT(stt.STT):
    """
    AWS Transcribe streaming Speech-to-Text implementation.

    Uses the smithy-based ``aws-sdk-transcribe-streaming`` client. Each
    "natural speech segment" detected by AWS is mapped to a turn:
    partials carry ``is_partial=True`` and the finalised segment arrives
    with ``is_partial=False``.

    Docs:
    - https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    - https://docs.aws.amazon.com/transcribe/latest/dg/streaming-partial-results.html
    """

    turn_detection: bool = True

    def __init__(
        self,
        language_code: str = "en-US",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_profile: Optional[str] = None,
        show_speaker_label: bool = False,
        enable_partial_results_stabilization: bool = False,
        partial_results_stability: Optional[str] = None,
    ):
        """
        Initialize AWS Transcribe streaming STT.

        Args:
            language_code: BCP-47 language code, e.g. ``"en-US"``.
            region_name: AWS region for the streaming endpoint.
            aws_access_key_id: Optional explicit access key.
            aws_secret_access_key: Optional explicit secret key.
            aws_session_token: Optional session token (for temporary creds).
            aws_profile: Optional named profile from ``~/.aws/credentials``.
                Resolved via boto3 to a static key/secret pair.
            show_speaker_label: Enable speaker diarization labels on items.
            enable_partial_results_stabilization: Stabilise the trailing words
                of partial transcripts to reduce flicker.
            partial_results_stability: ``"high"``, ``"medium"`` or ``"low"``.
                Only meaningful when stabilization is enabled.
        """
        super().__init__(provider_name="aws")

        self.language_code = language_code
        self.region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._aws_profile = aws_profile
        self.show_speaker_label = show_speaker_label
        self.enable_partial_results_stabilization = (
            enable_partial_results_stabilization
        )
        self.partial_results_stability = partial_results_stability

        self._client: Optional[TranscribeStreamingClient] = None
        self._stream: Optional[
            DuplexEventStream[
                AudioStream, TranscriptResultStream, StartStreamTranscriptionOutput
            ]
        ] = None
        self._input_stream: Optional[EventPublisher[AudioStream]] = None
        self._output_stream: Optional[EventReceiver[TranscriptResultStream]] = None
        self._recv_task: Optional[asyncio.Task[None]] = None
        self._current_participant: Optional[Participant] = None
        self._turn_in_progress = False
        self._audio_start_time: Optional[float] = None
        # Media-time watermarks, in seconds. Total audio fed into the stream
        # so far, and the cutoff snapshotted by clear(). Results whose
        # start_time precedes the watermark are suppressed. The lock
        # serialises increment-on-send with clear()'s snapshot so a chunk
        # mid-flight cannot escape the cutoff.
        self._audio_sent_seconds: float = 0.0
        self._start_time_watermark: float = 0.0
        self._watermark_lock = asyncio.Lock()

    async def start(self):
        await super().start()

        self._client = TranscribeStreamingClient(config=await self._build_config())

        stream = await asyncio.wait_for(
            self._client.start_stream_transcription(
                input=self._build_transcription_input()
            ),
            timeout=10.0,
        )
        self._stream = stream
        self._input_stream = stream.input_stream
        _, self._output_stream = await stream.await_output()
        self._recv_task = asyncio.create_task(self._recv_loop())

        logger.info(
            "AWS Transcribe streaming connection established (region=%s, lang=%s)",
            self.region_name,
            self.language_code,
        )

    async def _build_config(self) -> Config:
        kwargs: dict[str, Any] = {
            "region": self.region_name,
            "endpoint_uri": (
                f"https://transcribestreaming.{self.region_name}.amazonaws.com"
            ),
        }

        if self._aws_access_key_id and self._aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._aws_access_key_id
            kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            if self._aws_session_token:
                kwargs["aws_session_token"] = self._aws_session_token
        else:
            # boto3.Session() reads ~/.aws/credentials synchronously, so the
            # resolver has to be constructed off the event loop.
            kwargs["aws_credentials_identity_resolver"] = await asyncio.to_thread(
                Boto3CredentialsResolver, profile_name=self._aws_profile
            )

        return Config(**kwargs)

    def _build_transcription_input(self) -> StartStreamTranscriptionInput:
        kwargs: dict[str, Any] = {
            "language_code": self.language_code,
            "media_sample_rate_hertz": 16000,
            "media_encoding": "pcm",
        }
        if self.show_speaker_label:
            kwargs["show_speaker_label"] = True
        if self.enable_partial_results_stabilization:
            kwargs["enable_partial_results_stabilization"] = True
            if self.partial_results_stability is not None:
                kwargs["partial_results_stability"] = (
                    self.partial_results_stability
                )
        return StartStreamTranscriptionInput(**kwargs)

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Optional[Participant] = None,
    ):
        if self.closed or self._input_stream is None:
            return

        resampled = pcm_data.resample(16_000, 1)
        self._current_participant = participant
        if self._audio_start_time is None:
            self._audio_start_time = time.perf_counter()

        async with self._watermark_lock:
            await self._input_stream.send(
                AudioStreamAudioEvent(
                    value=AudioEvent(audio_chunk=resampled.samples.tobytes())
                )
            )
            self._audio_sent_seconds += resampled.duration

    async def _recv_loop(self):
        if self._output_stream is None:
            return
        try:
            async for event in self._output_stream:
                if isinstance(event, TranscriptResultStreamTranscriptEvent):
                    self._handle_transcript_event(event.value)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self.closed:
                logger.exception("AWS Transcribe receive loop failed")

    def _handle_transcript_event(self, event: TranscriptEvent):
        if event.transcript is None or not event.transcript.results:
            return

        participant = self._current_participant
        if participant is None:
            raise ValueError(
                "No participant set - audio must be processed with a participant"
            )

        for result in event.transcript.results:
            if result.start_time < self._start_time_watermark:
                continue
            if not result.alternatives:
                continue
            text = (result.alternatives[0].transcript or "").strip()
            if not text:
                continue

            response = self._result_to_response(result)

            if result.is_partial:
                if not self._turn_in_progress:
                    self._turn_in_progress = True
                    self._emit_turn_started_event(participant)
                self._emit_partial_transcript_event(text, participant, response)
            else:
                self._emit_transcript_event(text, participant, response)
                self._audio_start_time = None
                self._turn_in_progress = False
                self._emit_turn_ended_event(participant)

    def _result_to_response(self, result: Result) -> TranscriptResponse:
        items = result.alternatives[0].items or []
        scores = [i.confidence for i in items if i.confidence is not None]
        confidence = sum(scores) / len(scores) if scores else None

        other: dict[str, Any] = {
            "result_id": result.result_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
        }
        if items:
            other["items"] = [
                {
                    "type": item.type,
                    "content": item.content,
                    "start_time": item.start_time,
                    "end_time": item.end_time,
                    "speaker": item.speaker,
                    "confidence": item.confidence,
                    "stable": item.stable,
                }
                for item in items
            ]
        if result.channel_id:
            other["channel_id"] = result.channel_id

        processing_time_ms: Optional[float] = None
        if self._audio_start_time is not None:
            processing_time_ms = (
                time.perf_counter() - self._audio_start_time
            ) * 1000

        return TranscriptResponse(
            confidence=confidence,
            language=self.language_code,
            model_name="aws-transcribe-streaming",
            other=other,
            processing_time_ms=processing_time_ms,
        )

    async def clear(self):
        # AWS Transcribe has no native way to drop in-flight events. Snapshot
        # a media-time watermark so any result whose segment began before
        # this point is suppressed in _handle_transcript_event.
        await super().clear()
        async with self._watermark_lock:
            self._start_time_watermark = self._audio_sent_seconds
        self._audio_start_time = None
        self._turn_in_progress = False
        self._current_participant = None

    async def close(self):
        await super().close()

        self._audio_start_time = None

        if self._input_stream is not None:
            try:
                await self._input_stream.close()
            except Exception:
                logger.warning(
                    "Error closing AWS Transcribe input stream", exc_info=True
                )

        if self._recv_task is not None:
            await cancel_and_wait(self._recv_task)
            self._recv_task = None

        if self._stream is not None:
            try:
                await self._stream.close()
            except Exception:
                logger.warning(
                    "Error closing AWS Transcribe stream", exc_info=True
                )
            finally:
                self._stream = None

        self._input_stream = None
        self._output_stream = None
        self._client = None
