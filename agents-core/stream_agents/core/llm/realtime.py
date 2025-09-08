from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, Optional, TYPE_CHECKING, TypeVar

from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.audio_track import AudioStreamTrack
import asyncio
if TYPE_CHECKING:
    from stream_agents.core.agents import Agent

import abc
import logging
import uuid

from pyee.asyncio import AsyncIOEventEmitter
from av.dictionary import Dictionary

from ..events import (
    STSConnectedEvent,
    STSDisconnectedEvent,
    STSAudioInputEvent,
    STSAudioOutputEvent,
    STSTranscriptEvent,
    STSResponseEvent,
    STSConversationItemEvent,
    STSErrorEvent,
    PluginInitializedEvent,
    PluginClosedEvent,
    register_global_event,
)

T = TypeVar("T")


class RealtimeResponse(Generic[T]):
    def __init__(self, original: T, text: str):
        self.original = original
        self.text = text

BeforeCb = Callable[[List[Dictionary]], None]
AfterCb  = Callable[[RealtimeResponse], None]

logger = logging.getLogger(__name__)


class Realtime(AsyncIOEventEmitter, abc.ABC):
    """Base class for Realtime implementations.

    This abstract base class provides the foundation for implementing real-time
    speech-to-speech communication with AI agents. It handles event emission
    and connection state management.

    Key Features:
    - Event-driven architecture using AsyncIOEventEmitter
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

        logger.debug(
            "Initialized Realtime base class",
            extra={
                "session_id": self.session_id,
                "provider": self.provider_name,
            },
        )

        # Emit initialization event
        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
        )
        register_global_event(init_event)
        self.emit("initialized", init_event)

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
            self.output_track: AudioStreamTrack = AudioStreamTrack(
                framerate=24000, stereo=False, format="s16"
            )
        except Exception:  # pragma: no cover - allow providers to set later
            self.output_track = None  # type: ignore[assignment]

    @property
    def is_connected(self) -> bool:
        """Return True if the realtime session is currently active."""
        return self._is_connected

    @abc.abstractmethod
    async def connect(self):
        ...

    def attach_agent(self, agent: Agent):
        self.agent = agent
        self.before_response_listener = lambda x: agent.add_to_conversation(x)
        self.after_response_listener = lambda x: agent.after_response(x)

    def set_before_response_listener(self, before_response_listener: BeforeCb):
        self.before_response_listener = before_response_listener

    def set_after_response_listener(self, after_response_listener: AfterCb):
        self.after_response_listener = after_response_listener

    @abc.abstractmethod
    def send_audio_pcm(self, pcm: PcmData, target_rate: int = 48000):
        ...

    async def send_text(self, text: str):
        """Send a text message from the human side to the conversation.

        Providers should override to forward text upstream. Base implementation raises.
        """
        raise NotImplementedError("send_text must be implemented by Realtime providers")

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
        raise NotImplementedError("native_send_realtime_input is not implemented for this provider")

    async def simple_response(
        self,
        *,
        text: str,
        processors: Optional[List[Any]] = None,
        participant: Any = None,
        timeout: Optional[float] = 30.0,
    ) -> RealtimeResponse[Any]:
        """Send text and resolve when the assistant finishes the turn.

        Aggregates `STSResponseEvent` deltas (is_complete=False) until a final
        event with `is_complete=True` arrives, then returns the concatenated text.
        """
        # Notify before listener with a minimal normalized message shape
        try:
            normalized: List[Dictionary] = [Dictionary({"content": text, "role": "user"})]
            if hasattr(self, "before_response_listener"):
                self.before_response_listener(normalized)
        except Exception:
            # Do not fail if Dictionary creation is problematic
            pass

        collected_parts: List[str] = []
        done_fut: asyncio.Future[RealtimeResponse[Any]] = asyncio.get_event_loop().create_future()

        async def _on_response(event: Any):
            try:
                if isinstance(event, STSResponseEvent):
                    if event.text:
                        collected_parts.append(event.text)
                    if event.is_complete:
                        if not done_fut.done():
                            done_fut.set_result(RealtimeResponse(event, "".join(collected_parts)))
                        # remove listener once complete
                        self.remove_listener("response", _on_response)
            except Exception as e:  # pragma: no cover
                if not done_fut.done():
                    done_fut.set_exception(e)

        # Attach listener and send the text
        self.on("response", _on_response)  # type: ignore[arg-type]
        await self.send_text(text)

        # Wait for completion
        try:
            result = await asyncio.wait_for(done_fut, timeout=timeout)
        except asyncio.TimeoutError as e:
            # Cleanup listener on timeout
            self.remove_listener("response", _on_response)
            raise e

        # Notify after listener
        if hasattr(self, "after_response_listener"):
            await self.after_response_listener(result)

        return result
    
    def _emit_connected_event(self, session_config=None, capabilities=None):
        """Emit a structured connected event."""
        self._is_connected = True
        # Mark ready when connected if provider uses base emitter
        try:
            self._ready_event.set()
        except Exception:
            pass
        event = STSConnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            session_config=session_config,
            capabilities=capabilities,
        )
        register_global_event(event)
        self.emit("connected", event)  # Structured event

    def _emit_disconnected_event(self, reason=None, was_clean=True):
        """Emit a structured disconnected event."""
        self._is_connected = False
        event = STSDisconnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            reason=reason,
            was_clean=was_clean,
        )
        register_global_event(event)
        self.emit("disconnected", event)  # Structured event

    def _emit_audio_input_event(
        self, audio_data, sample_rate=16000, user_metadata=None
    ):
        """Emit a structured audio input event."""
        event = STSAudioInputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("audio_input", event)

    def _emit_audio_output_event(
        self, audio_data, sample_rate=16000, response_id=None, user_metadata=None
    ):
        """Emit a structured audio output event."""
        event = STSAudioOutputEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            response_id=response_id,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("audio_output", event)

    def _emit_transcript_event(
        self,
        text,
        is_user=True,
        confidence=None,
        conversation_item_id=None,
        user_metadata=None,
    ):
        """Emit a structured transcript event."""
        event = STSTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            is_user=is_user,
            confidence=confidence,
            conversation_item_id=conversation_item_id,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("transcript", event)

    def _emit_response_event(
        self,
        text,
        response_id=None,
        is_complete=True,
        conversation_item_id=None,
        user_metadata=None,
    ):
        """Emit a structured response event."""
        event = STSResponseEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            text=text,
            response_id=response_id,
            is_complete=is_complete,
            conversation_item_id=conversation_item_id,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("response", event)

    def _emit_conversation_item_event(
        self, item_id, item_type, status, role, content=None, user_metadata=None
    ):
        """Emit a structured conversation item event."""
        event = STSConversationItemEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            item_id=item_id,
            item_type=item_type,
            status=status,
            role=role,
            content=content,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("conversation_item", event)

    def _emit_error_event(self, error, context="", user_metadata=None):
        """Emit a structured error event."""
        event = STSErrorEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            error=error,
            context=context,
            user_metadata=user_metadata,
        )
        register_global_event(event)
        self.emit("error", event)  # Structured event

    async def close(self):
        """Close the Realtime service and release any resources."""
        if self._is_connected:
            await self._close_impl()
            self._emit_disconnected_event("service_closed", True)

        # Emit closure event
        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            cleanup_successful=True,
        )
        register_global_event(close_event)
        self.emit("closed", close_event)

    @abc.abstractmethod
    async def _close_impl(self):
        ...

# Public re-export
__all__ = ["Realtime"]
