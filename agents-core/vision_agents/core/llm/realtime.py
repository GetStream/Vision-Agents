import abc
import asyncio
import logging
import uuid
from collections.abc import Coroutine
from typing import (
    Any,
    Optional,
)

from getstream.video.rtc.track_util import PcmData
from vision_agents.core.edge.types import Participant

from . import OmniLLM, events

logger = logging.getLogger(__name__)


class Realtime(OmniLLM):
    """
    Realtime is an abstract base class for LLMs that can receive audio and video

    Example:

        llm = Realtime()
        llm.connect()
        llm.simple_response("what do you see?")

    Emits the following events:

    TODO: document/ evaluate how many events we want/ need...
        - Transcript incoming audio
        - Transcript outgoing audio

    """

    fps: int = 1
    session_id: str  # UUID to identify this session

    def __init__(
        self,
        fps: int = 1,  # the number of video frames per second to send (for implementations that support setting fps)
    ):
        super().__init__()
        self.connected = False

        self.provider_name = "realtime_base"
        self.session_id = str(uuid.uuid4())
        self.fps = fps
        # Store current participant for user speech transcription events
        self._current_participant: Optional[Participant] = None

        # Background tool tasks — tracked to prevent GC and awaited on close
        self._tool_tasks: set[asyncio.Task[None]] = set()

        # Monotonic epoch counter; incremented on interrupt so stale events
        # emitted before the interrupt can be identified and dropped.
        self._epoch: int = 0
        self._response_epoch: int = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def _begin_response(self) -> None:
        """Snapshot the current epoch for this response."""
        self._response_epoch = self._epoch

    async def interrupt(self) -> None:
        """Increment epoch so stale audio output events are discarded."""
        self._epoch += 1

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

    @abc.abstractmethod
    async def connect(self): ...

    @abc.abstractmethod
    async def simple_audio_response(
        self, pcm: PcmData, participant: Optional[Participant] = None
    ): ...

    async def stop_watching_video_track(self) -> None:
        """Optionally overridden by providers that support video input."""
        return None

    def _emit_connected_event(self, session_config=None, capabilities=None):
        """Emit a structured connected event."""
        self.connected = True
        # Mark ready when connected if provider uses base emitter
        try:
            self._ready_event.set()  # type: ignore[attr-defined]
        except Exception:
            pass
        event = events.RealtimeConnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            session_config=session_config,
            capabilities=capabilities,
        )
        self.events.send(event)

    def _emit_disconnected_event(self, reason=None, was_clean=True):
        """Emit a structured disconnected event."""
        self.connected = False
        event = events.RealtimeDisconnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            reason=reason,
            was_clean=was_clean,
        )
        self.events.send(event)

    def _emit_audio_input_event(self, audio_data: PcmData, user_metadata=None):
        """Emit a structured audio input event."""
        event = events.RealtimeAudioInputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            data=audio_data,
            participant=user_metadata,
        )
        self.events.send(event)

    # TODO: discussion around event vs output_track... why do we have both?
    def _emit_audio_output_event(
        self, audio_data: PcmData, response_id=None, user_metadata=None
    ):
        """Emit a structured audio output event."""
        event = events.RealtimeAudioOutputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            data=audio_data,
            response_id=response_id,
            epoch=self._response_epoch,
            participant=user_metadata,
        )
        self.events.send(event)

    def _emit_audio_output_done_event(
        self, response_id: str | None = None, user_metadata=None
    ):
        """Emit an event signaling audio output is complete."""
        event = events.RealtimeAudioOutputDoneEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            response_id=response_id,
            participant=user_metadata,
        )
        self.events.send(event)

    def _emit_response_event(
        self,
        text,
        response_id=None,
        is_complete=True,
        conversation_item_id=None,
        user_metadata=None,
    ):
        """Emit a structured response event."""
        event = events.RealtimeResponseEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            response_id=response_id,
            is_complete=is_complete,
            conversation_item_id=conversation_item_id,
            participant=user_metadata,
        )
        self.events.send(event)

    def _emit_conversation_item_event(
        self, item_id, item_type, status, role, content=None, user_metadata=None
    ):
        """Emit a structured conversation item event."""
        event = events.RealtimeConversationItemEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            item_id=item_id,
            item_type=item_type,
            status=status,
            role=role,
            content=content,
            participant=user_metadata,
        )
        self.events.send(event)

    def _emit_error_event(self, error, context="", user_metadata=None):
        """Emit a structured error event."""
        event = events.RealtimeErrorEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            error=error,
            context=context,
            participant=user_metadata,
        )
        self.events.send(event)

    @abc.abstractmethod
    async def close(self):
        raise NotImplementedError("llm.close isn't implemented")

    def _emit_user_speech_transcription(
        self,
        text: str,
        *,
        mode: events.TranscriptMode,
        original: Any = None,
    ):
        """Emit a user speech transcription event with participant info."""
        event = events.RealtimeUserSpeechTranscriptionEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            mode=mode,
            original=original,
            participant=self._current_participant,
        )
        self.events.send(event)

    def _emit_agent_speech_transcription(
        self,
        text: str,
        *,
        mode: events.TranscriptMode,
        original: Any = None,
    ):
        """Emit an agent speech transcription event."""
        event = events.RealtimeAgentSpeechTranscriptionEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            mode=mode,
            original=original,
        )
        self.events.send(event)
