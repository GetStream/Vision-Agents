"""
Realtime base class contract and streaming semantics
===================================================

Overview
--------
This module defines the provider-agnostic Realtime base used by Agent for
realtime integrations. Providers (e.g., Gemini Live, OpenAI
Realtime) implement how to send input (text/audio/video) and emit standardized
events. The base class aggregates streamed responses and exposes a consistent
API to users and the Agent.

Events that providers must emit
-------------------------------
- RealtimeResponseEvent (EventType.REALTIME_RESPONSE)
  - is_complete=False: a delta/partial textual update during streaming
  - is_complete=True: a single end-of-turn "done" signal; text may be present
    (some providers repeat the last chunk) or empty (audio-only turns)
- RealtimeTranscriptEvent for user/assistant transcripts (optional but recommended)
- RealtimeAudioOutputEvent for audio bytes returned by the model
- RealtimeAudioInputEvent when audio is sent upstream (optional)

Aggregation contract (hybrid strategy)
-------------------------------------
Both simple_response and native_response aggregate text using a hybrid approach:
1) Concatenate only deltas (is_complete=False) into an accumulator
2) When the done event (is_complete=True) arrives:
   - If no deltas were seen, use the done text (handles providers that only
     send final text)
   - If deltas were seen, prefer the accumulated text but:
     - If done text starts with the accumulated text, use the done text to
       preserve final punctuation/formatting
     - Else, if done text is a short punctuation-only suffix, append it
     - Else, keep the accumulated text

Return type
-----------
Both simple_response and native_response return RealtimeResponse:
- .text: the aggregated final text following the rules above
- .original: the final RealtimeResponseEvent (is_complete=True) that ended the turn

Notes
-----
- Providers should emit exactly one done event per turn
- For audio-only turns, done should be emitted with empty text; callers receive
  RealtimeResponse(text="") but the audio is still emitted on the "audio" event
"""

from __future__ import annotations

import pprint
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Awaitable,
)

from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.audio_track import AudioStreamTrack
import asyncio

from ..utils.utils import parse_instructions

if TYPE_CHECKING:
    from stream_agents.core.agents import Agent

import abc
import logging
import uuid

from stream_agents.core.events import PluginInitializedEvent, PluginClosedEvent
from stream_agents.core.events.manager import EventManager

from . import events

T = TypeVar("T")


class RealtimeResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text


BeforeCb = Callable[[List[Any]], None]
AfterCb = Callable[[RealtimeResponse[Any]], Any]

logger = logging.getLogger(__name__)


class Realtime(abc.ABC):
    """Base class for Realtime implementations.

    This abstract base class provides the foundation for implementing real-time
    speech-to-speech communication with AI agents. It handles event emission
    and connection state management.

    Key Features:
    - Connection state tracking
    - Standardized event interface

    Implementations should:
    1. Establish and manage the audio session
    2. Handle provider-specific authentication and setup
    3. Emit appropriate events for state changes and interactions
    4. Implement any provider-specific helper methods
    """

    def __init__(
        self,
        *,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        temperature: Optional[float] = None,
        voice: Optional[str] = None,
        provider_config: Optional[Any] = None,
        response_modalities: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **_: Any,
    ):
        """Initialize base Realtime class with common, provider-agnostic preferences.

        These fields are optional hints that concrete providers may choose to map
        to their own session/config structures. They are not enforced here.

        Args:
            provider_name: Optional provider name override. Defaults to class name.
            model: Model ID to use when connecting.
            instructions: Optional system instructions passed to the session.
            temperature: Optional temperature passed to the session.
            voice: Optional voice selection passed to the session.
            provider_config: Provider-specific configuration (e.g., Gemini Live config, OpenAI session prefs).
            response_modalities: Optional response modalities passed to the session.
            tools: Optional tools passed to the session.
        """
        super().__init__()
        self._is_connected = False
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        # Ready event for providers to signal readiness
        self._ready_event: asyncio.Event = asyncio.Event()
        self.events = EventManager()
        self.events.register_events_from_module(events)
        # Common, optional preferences (not all providers will use all of these)
        self.model = model
        self.instructions = instructions
        self.temperature = temperature
        self.voice = voice
        # Provider-specific configuration (e.g., Gemini Live config, OpenAI session prefs)
        self.provider_config = provider_config
        self.response_modalities = response_modalities
        self.tools = tools
        # Default outbound audio track for assistant speech; providers can override
        try:
            # Use 48000 Hz stereo to match common WebRTC Opus SDP (opus/48000/2)
            self.output_track: AudioStreamTrack = AudioStreamTrack(
                framerate=48000, stereo=True, format="s16"
            )
        except Exception:  # pragma: no cover - allow providers to set later
            self.output_track = None  # type: ignore[assignment]

        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
        )
        self.events.append(init_event)


    @property
    def is_connected(self) -> bool:
        """Return True if the realtime session is currently active."""
        return self._is_connected

    def _attach_agent(self, agent: Agent):
        """
        Attach agent to the llm
        """
        self.agent = agent
        self._conversation = agent.conversation
        self.instructions = agent.instructions

        # Parse instructions to extract @ mentioned markdown files
        self.parsed_instructions = parse_instructions(agent.instructions)

    @abc.abstractmethod
    async def connect(self): ...

    # @abc.abstractmethod
    # async def send_audio_pcm(self, pcm: PcmData, target_rate: int = 48000): ...

    @abc.abstractmethod
    async def send_text(self, text: str):
        """Send a text message from the human side to the conversation.

        Providers should override to forward text upstream.
        """
        ...

    async def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the realtime session is ready. Returns True if ready."""
        if self._ready_event.is_set():
            return True
        try:
            return await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    async def start_video_sender(self, track: Any, fps: int = 1) -> None:
        """Optionally overridden by providers that support video input."""
        return None

    async def stop_video_sender(self) -> None:
        """Optionally overridden by providers that support video input."""
        return None

    async def interrupt_playback(self) -> None:
        """Optionally overridden by providers to stop current audio playback."""
        return None

    def resume_playback(self) -> None:
        """Optionally overridden by providers to resume audio playback."""
        return None

    # --- Optional provider-native passthroughs for advanced usage ---
    def get_native_session(self) -> Any:
        """Return underlying provider session if available (advanced use).

        Providers should override to return their native session object.
        Default returns None.
        """
        return None

    async def native_send_realtime_input(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
    ) -> None:
        """Advanced: provider-native realtime input (text/audio/media).

        Providers that support a native realtime input API should override this.
        Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            "native_send_realtime_input is not implemented for this provider"
        )

    async def native_response(
        self,
        *,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        media: Optional[Any] = None,
        timeout: Optional[float] = 30.0,
    ) -> RealtimeResponse[Any]:
        """Provider-native request that returns a standardized RealtimeResponse.

        Delegates to the shared aggregation helper using native_send_realtime_input.
        """

        async def _sender():
            await self.native_send_realtime_input(text=text, audio=audio, media=media)

        return await self._aggregate_turn(
            sender=_sender, before_text=text, timeout=timeout
        )

    async def simple_response(
        self,
        *,
        text: str,
        timeout: Optional[float] = 30.0,
    ) -> RealtimeResponse[Any]:
        """Send text and resolve when the assistant finishes the turn.

        Aggregates streaming deltas with a hybrid strategy to build final text.
        """

        async def _sender():
            await self.send_text(text)

        return await self._aggregate_turn(
            sender=_sender, before_text=text, timeout=timeout
        )

    # ---- Shared aggregation helpers ----
    @staticmethod
    def _merge_final_text(collected_parts: List[str], done_text: Optional[str]) -> str:
        accumulated = "".join(collected_parts)
        final_done = done_text or ""
        if not collected_parts:
            return final_done
        if final_done.startswith(accumulated):
            return final_done
        if accumulated.endswith(final_done):
            return accumulated
        if (
            final_done
            and len(final_done) <= 4
            and all(ch in " .,!?:;—-…'\"" for ch in final_done)
        ):
            return accumulated + final_done
        return accumulated

    async def _aggregate_turn(
        self,
        *,
        sender: Callable[[], Awaitable[None]],
        before_text: Optional[str],
        timeout: Optional[float],
    ) -> RealtimeResponse[Any]:
        # Notify before listener
        if before_text is not None:
            try:
                normalized: List[Any] = [{"content": before_text, "role": "user"}]
                if hasattr(self, "before_response_listener"):
                    self.before_response_listener(normalized)
            except Exception:
                pass

        collected_parts: List[str] = []

        async def _on_response(event: events.RealtimeResponseEvent):
            if event.is_complete:
                final_text = self._merge_final_text(
                    collected_parts, event.text
                )
                collected_parts = []
                if hasattr(self, "after_response_listener"):
                    await self.after_response_listener(RealtimeResponse(event, final_text))
            else:
                if event.text:
                    collected_parts.append(event.text)

        self.events.subscribe(_on_response)  # type: ignore[arg-type]
        await sender()

        return result

    def _emit_connected_event(self, session_config=None, capabilities=None):
        """Emit a structured connected event."""
        self._is_connected = True
        # Mark ready when connected if provider uses base emitter
        try:
            self._ready_event.set()
        except Exception:
            pass
        event = events.RealtimeConnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            session_config=session_config,
            capabilities=capabilities,
        )
        self.events.append(event)

    def _emit_disconnected_event(self, reason=None, was_clean=True):
        """Emit a structured disconnected event."""
        self._is_connected = False
        event = events.RealtimeDisconnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            reason=reason,
            was_clean=was_clean,
        )
        self.events.append(event)

    def _emit_audio_input_event(
        self, audio_data, sample_rate=16000, user_metadata=None
    ):
        """Emit a structured audio input event."""
        event = events.RealtimeAudioInputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            user_metadata=user_metadata,
        )
        self.events.append(event)

    def _emit_audio_output_event(
        self, audio_data, sample_rate=16000, response_id=None, user_metadata=None
    ):
        """Emit a structured audio output event."""
        event = events.RealtimeAudioOutputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            response_id=response_id,
            user_metadata=user_metadata,
        )
        self.events.append(event)

    def _emit_partial_transcript_event(self, text: str, user_metadata=None, original=None):
        event = events.RealtimeTranscriptEvent(
            text=text,
            user_metadata=user_metadata,
            original=original,
        )
        self.events.append(event)

    def _emit_transcript_event(
        self,
        text: str,
        user_metadata=None,
        original=None,
    ):
        event = events.RealtimeTranscriptEvent(
            text=text,
            user_metadata=user_metadata,
            original=original,
        )
        self.events.append(event)

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
            user_metadata=user_metadata,
        )
        self.events.append(event)

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
            user_metadata=user_metadata,
        )
        self.events.append(event)

    def _emit_error_event(self, error, context="", user_metadata=None):
        """Emit a structured error event."""
        event = events.RealtimeErrorEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            error=error,
            context=context,
            user_metadata=user_metadata,
        )
        self.events.append(event)

    async def close(self):
        """Close the Realtime service and release any resources."""
        if self._is_connected:
            await self._close_impl()
            self._emit_disconnected_event("service_closed", True)

        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            cleanup_successful=True,
        )
        self.events.append(close_event)

    @abc.abstractmethod
    async def _close_impl(self): ...

