import abc
import logging
import uuid

from typing import Any, Dict, List, Optional

from pyee.asyncio import AsyncIOEventEmitter

from .events import (
    STSConnectedEvent, STSDisconnectedEvent, STSAudioInputEvent, STSAudioOutputEvent,
    STSTranscriptEvent, STSResponseEvent, STSConversationItemEvent, STSErrorEvent,
    PluginInitializedEvent, PluginClosedEvent
)
from .event_utils import register_global_event

logger = logging.getLogger(__name__)


class STS(AsyncIOEventEmitter, abc.ABC):
    """Base class for Speech-to-Speech (STS) implementations.

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
        """Initialize base STS with common, provider-agnostic preferences.

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

        logger.debug(
            "Initialized STS base class",
            extra={
                "session_id": self.session_id,
                "provider": self.provider_name,
            },
        )

        # Emit initialization event
        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STS",
            provider=self.provider_name,
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

    @property
    def is_connected(self) -> bool:
        """Return True if the realtime session is currently active."""
        return self._is_connected

    def _emit_connected_event(self, session_config=None, capabilities=None):
        """Emit a structured connected event."""
        self._is_connected = True
        event = STSConnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            provider=self.provider_name,
            session_config=session_config,
            capabilities=capabilities
        )
        register_global_event(event)
        self.emit("connected", event)  # Structured event

    def _emit_disconnected_event(self, reason=None, was_clean=True):
        """Emit a structured disconnected event."""
        self._is_connected = False
        event = STSDisconnectedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            provider=self.provider_name,
            reason=reason,
            was_clean=was_clean
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
            plugin_type="STS",
            provider=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            user_metadata=user_metadata
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
            plugin_type="STS",
            provider=self.provider_name,
            audio_data=audio_data,
            sample_rate=sample_rate,
            response_id=response_id,
            user_metadata=user_metadata
        )
        register_global_event(event)
        self.emit("audio_output", event)

    def _emit_transcript_event(
        self, text, is_user=True, confidence=None,
        conversation_item_id=None, user_metadata=None
    ):
        """Emit a structured transcript event."""
        event = STSTranscriptEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STS",
            provider=self.provider_name,
            text=text,
            is_user=is_user,
            confidence=confidence,
            conversation_item_id=conversation_item_id,
            user_metadata=user_metadata
        )
        register_global_event(event)
        self.emit("transcript", event)

    def _emit_response_event(
        self, text, response_id=None, is_complete=True,
        conversation_item_id=None, user_metadata=None
    ):
        """Emit a structured response event."""
        event = STSResponseEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STS",
            provider=self.provider_name,
            text=text,
            response_id=response_id,
            is_complete=is_complete,
            conversation_item_id=conversation_item_id,
            user_metadata=user_metadata
        )
        register_global_event(event)
        self.emit("response", event)

    def _emit_conversation_item_event(
        self, item_id, item_type, status, role,
        content=None, user_metadata=None
    ):
        """Emit a structured conversation item event."""
        event = STSConversationItemEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STS",
            provider=self.provider_name,
            item_id=item_id,
            item_type=item_type,
            status=status,
            role=role,
            content=content,
            user_metadata=user_metadata
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
            user_metadata=user_metadata
        )
        register_global_event(event)
        self.emit("error", event)  # Structured event

    async def close(self):
        """Close the STS service and release any resources."""
        if self._is_connected:
            self._emit_disconnected_event("service_closed", True)

        # Emit closure event
        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="STS",
            provider=self.provider_name,
            cleanup_successful=True
        )
        register_global_event(close_event)
        self.emit("closed", close_event)


# Public re-export
__all__ = ["STS"]
