import abc
import asyncio
import logging
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.transcript import TranscriptMode
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm import OmniLLM
from vision_agents.core.llm.events import (
    LLMErrorEvent,
    RealtimeConnectedEvent,
    RealtimeDisconnectedEvent,
)
from vision_agents.core.utils.stream import Stream

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RealtimeAudioOutput:
    data: PcmData
    response_id: str | None = None


@dataclass(slots=True)
class RealtimeAudioOutputDone:
    """Event emitted when audio output generation is complete for a response."""

    interrupted: bool = False
    response_id: str | None = None


@dataclass(slots=True)
class RealtimeUserTranscript:
    participant: Participant
    mode: TranscriptMode
    text: str = ""


@dataclass(slots=True)
class RealtimeAgentTranscript:
    mode: TranscriptMode
    text: str = ""


@dataclass(slots=True)
class RealtimeUserSpeechStarted:
    """Server-side VAD signal: the user started speaking."""

    participant: Participant


@dataclass(slots=True)
class RealtimeUserSpeechEnded:
    """Server-side VAD signal: the user stopped speaking."""

    participant: Participant


@dataclass(slots=True)
class RealtimeAgentSpeechStarted:
    """The model started producing audio output for a response."""

    response_id: str | None = None


@dataclass(slots=True)
class RealtimeAgentSpeechEnded:
    """The model finished producing audio output for a response."""

    interrupted: bool = False
    response_id: str | None = None


class Realtime(OmniLLM):
    """
    Realtime is an abstract base class for LLMs that can receive audio and video

    Example:

        llm = Realtime()
        llm.connect()
        llm.simple_response("what do you see?")

    """

    fps: int = 1
    session_id: str  # UUID to identify this session

    def __init__(
        self,
        fps: int = 1,  # the number of video frames per second to send (for implementations that support setting fps)
    ):
        super().__init__()
        self.connected = False

        self.session_id = str(uuid.uuid4())
        self.fps = fps
        # Store current participant for user speech transcription events
        self._current_participant: Participant | None = None

        # Background tool tasks — tracked to prevent GC and awaited on close
        self._tool_tasks: set[asyncio.Task[None]] = set()

        # Monotonic epoch counter; incremented on interrupt so stale events
        # emitted before the interrupt can be identified and dropped.
        self._epoch: int = 0
        self._output: Stream[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
            | RealtimeUserSpeechStarted
            | RealtimeUserSpeechEnded
            | RealtimeAgentSpeechStarted
            | RealtimeAgentSpeechEnded
        ] = Stream()

    @property
    def output(
        self,
    ) -> Stream[
        RealtimeAudioOutput
        | RealtimeAudioOutputDone
        | RealtimeUserTranscript
        | RealtimeAgentTranscript
        | RealtimeUserSpeechStarted
        | RealtimeUserSpeechEnded
        | RealtimeAgentSpeechStarted
        | RealtimeAgentSpeechEnded
    ]:
        """Pipeline output stream: consumers iterate, subclasses push via send_nowait."""
        return self._output

    @abc.abstractmethod
    async def connect(self): ...

    @abc.abstractmethod
    async def simple_audio_response(self, pcm: PcmData, participant: Participant): ...

    @abc.abstractmethod
    async def close(self): ...

    @property
    def epoch(self) -> int:
        return self._epoch

    async def interrupt(self) -> None:
        """Increment epoch so stale audio output events are discarded."""
        self._epoch += 1
        self._current_participant = None
        self._output.clear()

    def _run_tool_in_background(self, coro: Coroutine[None, None, None]) -> None:
        """Run a tool coroutine as a background task without blocking the WS reader."""
        task = asyncio.create_task(coro)
        self._tool_tasks.add(task)
        task.add_done_callback(self._on_tool_task_done)

    def _on_tool_task_done(self, task: asyncio.Task[None]) -> None:
        """Callback for completed tool tasks — log exceptions and clean up."""
        self._tool_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.exception("Background tool task failed", exc_info=task.exception())

    async def _await_pending_tools(self) -> None:
        """Await all in-flight tool tasks. Call this in close() before closing the connection."""
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
            self._tool_tasks.clear()

    async def process_audio(self, pcm: PcmData, participant: Participant) -> None:
        self._current_participant = participant
        await self.simple_audio_response(pcm, participant)

    async def stop_watching_video_track(self) -> None:
        """Optionally overridden by providers that support video input."""
        ...

    def _on_connected(
        self,
        session_config: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
    ):
        """Mark the session connected and emit RealtimeConnectedEvent."""
        self.connected = True
        self.events.send(
            RealtimeConnectedEvent(
                plugin_name=self.provider_name,
                session_id=self.session_id,
                session_config=session_config,
                capabilities=capabilities,
            )
        )

    def _on_disconnected(self, reason: str | None = None, clean: bool = True):
        """Mark the session disconnected and emit RealtimeDisconnectedEvent."""
        self.connected = False
        self.events.send(
            RealtimeDisconnectedEvent(
                plugin_name=self.provider_name,
                session_id=self.session_id,
                reason=reason,
                clean=clean,
            )
        )

    def _emit_audio_output_event(self, pcm: PcmData, response_id: str | None = None):
        """Emit a structured audio output event."""
        event = RealtimeAudioOutput(
            data=pcm,
            response_id=response_id,
        )
        self._output.send_nowait(event)
        self.metrics.on_realtime_audio_output(
            byte_count=pcm.samples.nbytes,
            duration_ms=pcm.duration_ms,
            provider=self.provider_name,
        )

    def _emit_audio_output_done_event(
        self,
        response_id: str | None = None,
        interrupted: bool = False,
    ):
        """Emit an event signaling audio output is complete."""
        event = RealtimeAudioOutputDone(
            response_id=response_id, interrupted=interrupted
        )
        self._output.send_nowait(event)

    def _emit_user_speech_started(self):
        """Emit a user-speech-started signal from server-side VAD."""
        if self._current_participant is None:
            return
        self._output.send_nowait(
            RealtimeUserSpeechStarted(participant=self._current_participant)
        )

    def _emit_user_speech_ended(self):
        """Emit a user-speech-ended signal from server-side VAD."""
        if self._current_participant is None:
            return
        self._output.send_nowait(
            RealtimeUserSpeechEnded(participant=self._current_participant)
        )

    def _emit_agent_speech_started(self, response_id: str | None = None):
        """Emit an agent-speech-started signal when the model begins audio output."""
        self._output.send_nowait(RealtimeAgentSpeechStarted(response_id=response_id))

    def _emit_agent_speech_ended(
        self, response_id: str | None = None, interrupted: bool = False
    ):
        """Emit an agent-speech-ended signal when the model stops producing audio."""
        self._output.send_nowait(
            RealtimeAgentSpeechEnded(response_id=response_id, interrupted=interrupted)
        )

    def _emit_user_speech_transcription(
        self,
        text: str,
        *,
        mode: TranscriptMode,
    ):
        """Emit a user speech transcription event with participant info."""
        # _current_participant can be None when the response is interrupted.
        if self._current_participant is not None:
            event = RealtimeUserTranscript(
                text=text,
                mode=mode,
                participant=self._current_participant,
            )
            self._output.send_nowait(event)
            self.metrics.on_realtime_user_transcription(provider=self.provider_name)

    def _emit_agent_speech_transcription(self, text: str, *, mode: TranscriptMode):
        """Emit an agent speech transcription event."""
        event = RealtimeAgentTranscript(text=text, mode=mode)
        self._output.send_nowait(event)
        self.metrics.on_realtime_agent_transcription(provider=self.provider_name)

    def _emit_response_event(
        self,
        text,
        response_id=None,
        is_complete=True,
        conversation_item_id=None,
        user_metadata=None,
    ):
        """Record metrics for a completed response."""
        if is_complete:
            self.metrics.on_realtime_response_completed(provider=self.provider_name)

    def _emit_error_event(self, error, context="", user_metadata=None):
        """Record metrics and emit LLMErrorEvent for a model error."""
        self.metrics.on_realtime_error(
            provider=self.provider_name,
            error_type=type(error).__name__ if error is not None else None,
        )
        self.events.send(
            LLMErrorEvent(
                plugin_name=self.provider_name,
                error=error,
                context=context,
            )
        )
