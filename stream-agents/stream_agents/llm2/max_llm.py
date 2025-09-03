import abc
import logging
import uuid
from collections.abc import Sequence
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
)

from pyee.asyncio import AsyncIOEventEmitter

from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData

from .event_utils import register_global_event
from .events import (
    ConnectionState,
    LLMConnectionEvent,
    LLMErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    LLMStreamDeltaEvent,
    PluginClosedEvent,
    PluginInitializedEvent,
)
from .llm_types import (
    ImageBytesPart,
    ImageURLPart,
    Message,
    NormalizedAudioItem,
    NormalizedOutputItem,
    NormalizedResponse,
    ResponseFormat,
    Role,
)

# Provider type variables for generic LLM base (unbounded; provider-defined)
TMessage = TypeVar("TMessage")
TTool = TypeVar("TTool")
TOptions = TypeVar("TOptions")
TSyncResult = TypeVar("TSyncResult")
TStreamItem = TypeVar("TStreamItem")

logger = logging.getLogger(__name__)


class LLM(
    AsyncIOEventEmitter,
    abc.ABC,
    Generic[TMessage, TTool, TOptions, TSyncResult, TStreamItem],
):
    """Base class for multimodal LLM providers.

    Responsibilities:
    - Normalize requests (messages, tools, options)
    - Emit structured events (request, delta, response, error, usage in response payload)
    - Provide sync and streaming APIs via abstract provider hooks

    Notes:
    - This class is generic over provider-specific types for messages, tools, options,
      sync results, and stream items.
    - `response_format` allows callers to request structured outputs (e.g., JSON Schema).
    - `metadata` is a free-form dict for traceability and routing; it is forwarded to
      events and provider hooks unchanged.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        client: Optional[Any] = None,
        **_: Any,
    ):
        super().__init__()
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__
        self.model = model
        self.client: Optional[Any] = client

        # Emit initialization event
        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="LLM",
            provider=self.provider_name,
            configuration={"model": self.model} if self.model else None,
        )
        register_global_event(init_event)
        self.emit("initialized", init_event)

    async def create_response(
        self,
        messages: Sequence[TMessage],
        *,
        tools: Optional[Sequence[TTool]] = None,
        options: Optional[TOptions] = None,
        response_format: Optional[ResponseFormat] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TSyncResult:
        """Request a non-streaming response.

        Emits LLM_REQUEST and then either LLM_RESPONSE or LLM_ERROR.

        Parameters:
        - messages: Provider-specific message objects. Read-only sequence.
        - tools: Provider-specific tool definitions, if any. Read-only sequence.
        - options: Provider-specific request options.
        - response_format: Optional structured-output request (e.g., JSON Schema).
        - metadata: Optional per-request metadata passed through to events/providers.

        Returns:
        - Provider-specific sync result (TSyncResult). A corresponding LLM_RESPONSE
          event is emitted with `raw` set to this result; providers may also supply
          a normalized response in future adapters.
        """
        req_event = LLMRequestEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            model_name=self.model,
            messages=messages,
            tools=tools,
            options=options,
            response_format=response_format,
            metadata=metadata,
        )
        register_global_event(req_event)
        self.emit("request", req_event)

        try:
            result = await self._create_response_impl(
                messages=messages,
                tools=tools,
                options=options,
                response_format=response_format,
                metadata=metadata,
            )

            normalized = self._normalize_sync_result_payload(result)
            resp_event = LLMResponseEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                normalized_response=normalized,
                raw=result,
            )
            register_global_event(resp_event)
            self.emit("response", resp_event)

            return result
        except Exception as e:
            err_event = LLMErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=e,
                context="create_response",
            )
            register_global_event(err_event)
            self.emit("error", err_event)
            raise

    async def create_response_stream(
        self,
        messages: Sequence[TMessage],
        *,
        tools: Optional[Sequence[TTool]] = None,
        options: Optional[TOptions] = None,
        response_format: Optional[ResponseFormat] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[TStreamItem]:
        """Request a streaming response.

        Emits LLM_REQUEST, then a sequence of LLM_STREAM_DELTA events, and a final
        LLM_RESPONSE. Yields provider-specific stream items (TStreamItem).

        Parameters:
        - messages: Provider-specific message objects. Read-only sequence.
        - tools: Provider-specific tool definitions, if any. Read-only sequence.
        - options: Provider-specific request options.
        - response_format: Optional structured-output request (e.g., JSON Schema).
        - metadata: Optional per-request metadata passed through to events/providers.

        Notes:
        - Stream deltas are forwarded in `LLMStreamDeltaEvent.delta`. Adapters may also
          populate `normalized_delta` for OpenAI Responses–style output items.
        - Convenience text/audio delta fields can be added later without breaking this API.
        """
        req_event = LLMRequestEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            model_name=self.model,
            messages=messages,
            tools=tools,
            options=options,
            response_format=response_format,
            metadata=metadata,
        )
        register_global_event(req_event)
        self.emit("request", req_event)

        iterator = await self._create_response_stream_impl(
            messages=messages,
            tools=tools,
            options=options,
            response_format=response_format,
            metadata=metadata,
        )
        async for delta in iterator:
            normalized_delta = self._normalize_stream_item_payload(delta)
            delta_event = LLMStreamDeltaEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                delta=delta,
                normalized_delta=normalized_delta,
                event_name=(
                    f"{normalized_delta.get('type')}.delta"
                    if isinstance(normalized_delta, dict) and "type" in normalized_delta
                    else None
                ),
                is_final=bool(delta.get("is_final"))
                if isinstance(delta, dict)
                else False,
            )
            register_global_event(delta_event)
            self.emit("delta", delta_event)

            # Convenience text delta event
            text_delta = self._extract_text_delta(delta)
            if isinstance(text_delta, str) and text_delta:
                text_event = LLMStreamDeltaEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    delta=None,
                    normalized_delta={"type": "text", "text": text_delta},
                    event_name="text.delta",
                    is_final=False,
                )
                register_global_event(text_event)
                self.emit("delta", text_event)

            # Convenience audio delta event
            audio_info = self._extract_audio_delta(delta)
            if isinstance(audio_info, dict) and audio_info.get("data"):
                data_val = audio_info.get("data")
                if isinstance(data_val, (bytes, bytearray)):
                    audio_item: NormalizedAudioItem = {
                        "type": "audio",
                        "data": bytes(data_val),
                    }
                    mime_val = audio_info.get("mime_type")
                    if isinstance(mime_val, str):
                        audio_item["mime_type"] = mime_val
                    sr_val = audio_info.get("sample_rate")
                    if isinstance(sr_val, int):
                        audio_item["sample_rate"] = sr_val
                    ch_val = audio_info.get("channels")
                    if isinstance(ch_val, int):
                        audio_item["channels"] = ch_val

                    audio_event = LLMStreamDeltaEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        delta=None,
                        normalized_delta=audio_item,
                        event_name="audio.delta",
                        is_final=False,
                    )
                    register_global_event(audio_event)
                    self.emit("delta", audio_event)
            yield delta

    @abc.abstractmethod
    async def _create_response_impl(
        self,
        *,
        messages: Sequence[TMessage],
        tools: Optional[Sequence[TTool]],
        options: Optional[TOptions],
        response_format: Optional[ResponseFormat],
        metadata: Optional[Dict[str, Any]],
    ) -> TSyncResult:
        """Provider-specific non-streaming implementation.

        Implementations should return a provider-specific result (TSyncResult) and
        respect `response_format` and `metadata` where supported by the provider.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _create_response_stream_impl(
        self,
        *,
        messages: Sequence[TMessage],
        tools: Optional[Sequence[TTool]],
        options: Optional[TOptions],
        response_format: Optional[ResponseFormat],
        metadata: Optional[Dict[str, Any]],
    ) -> AsyncIterator[TStreamItem]:
        """Provider-specific streaming implementation.

        Implementations should yield provider-specific stream items (TStreamItem) and
        respect `response_format` and `metadata` where supported by the provider.
        A final response event will typically be emitted by the provider when the
        stream ends (if available).
        """
        raise NotImplementedError

    # =============================
    # Normalization / Convenience hooks
    # =============================

    # def _map_turn_input(
    #     self,
    #     *,
    #     text: Optional[str],
    #     inputs: Optional[List[ContentPart]],
    #     state: Optional[Dict[str, Any]],
    # ) -> List[Message]:
    #     """Default mapper from a simple turn input to a `Message` list.

    #     - text: user text input
    #     - inputs: optional additional content parts (image/audio/etc.)
    #     - state: optional processor state; mapped into a {"type": "json", "data": ...} part
    #     """
    #     content: List[ContentPart] = []
    #     if text:
    #         content.append({"type": "text", "text": text})
    #     if inputs:
    #         content.extend(inputs)
    #     if state:
    #         content.append({"type": "json", "data": state})
    #     if not content:
    #         raise ValueError("At least one of text, inputs, or state must be provided")
    #     return [
    #         {
    #             "role": Role.USER,
    #             "content": content,
    #         }
    #     ]

    # TODO: make these abstract
    def _normalize_sync_result_payload(
        self, result: TSyncResult
    ) -> Optional[NormalizedResponse]:
        """Optional: Map provider sync result to NormalizedResponse.

        Default: return None. Providers can override to supply normalized payload.
        """
        return None

    def _normalize_stream_item_payload(
        self, item: TStreamItem
    ) -> Optional[NormalizedOutputItem]:
        """Optional: Map provider stream item to a NormalizedOutputItem.

        Default: return None. Providers can override.
        """
        return None

    def _extract_text_delta(self, item: TStreamItem) -> Optional[str]:
        """Optional: Extract plain text delta for convenience text events.

        Default: return None. Providers can override.
        """
        return None

    def _extract_audio_delta(self, item: TStreamItem) -> Optional[Dict[str, Any]]:
        """Optional: Extract audio chunk for convenience audio events.

        Should return a dict with keys: data (bytes) and optionally
        mime_type (str), sample_rate (int), channels (int). Default: None.
        """
        return None

    async def close(self):
        """Close the LLM and release any resources."""
        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="LLM",
            provider=self.provider_name,
            cleanup_successful=True,
        )
        register_global_event(close_event)
        self.emit("closed", close_event)


class RealtimeLLM(LLM[TMessage, TTool, TOptions, TSyncResult, TStreamItem]):
    """Base class for realtime-capable LLM providers.

    Adds connection lifecycle and event emission. Concrete classes should
    manage the underlying session and forward provider events into the LLM events.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._is_connected = False
        self._output_track: Optional[AudioStreamTrack] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @abc.abstractmethod
    async def connect(self, *args: Any, **kwargs: Any) -> Any:
        """Establish a realtime session.

        Guidance on state events:
        - Emit ConnectionState.CONNECTING before network activity, then CONNECTED on success
        - Emit RECONNECTING on transient reconnect attempts
        - Emit ERROR on unrecoverable failures (and then DISCONNECTED)
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def _disconnect_impl(self, reason: Optional[str] = None) -> None:
        """Provider-specific disconnect (close sockets, tasks, etc.)."""
        raise NotImplementedError

    async def disconnect(self, reason: Optional[str] = None) -> None:
        """Disconnect the realtime session and emit a DISCONNECTED connection event."""
        await self._disconnect_impl(reason)
        self._emit_connection(
            ConnectionState.DISCONNECTED, {"reason": reason} if reason else None
        )

    def _emit_connection(
        self, state: ConnectionState, details: Optional[Dict[str, Any]] = None
    ) -> None:
        self._is_connected = state == ConnectionState.CONNECTED
        event = LLMConnectionEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            connection_state=state,
            details=details,
        )
        register_global_event(event)
        self.emit("connection", event)

    # Realtime helpers
    def set_output_track(self, track: AudioStreamTrack) -> None:
        """Set an audio output track for providers that return audio in realtime.

        Providers should write audio bytes to `self.output_track.write(...)` when producing audio.
        """
        self._output_track = track

    def create_output_track(
        self, *, framerate: int = 24000, stereo: bool = False, format: str = "s16"
    ) -> AudioStreamTrack:
        """Create and set a default output audio track and return it."""
        track = AudioStreamTrack(framerate=framerate, stereo=stereo, format=format)
        self.set_output_track(track)
        return track

    @property
    def output_track(self) -> Optional[AudioStreamTrack]:
        return self._output_track

    @abc.abstractmethod
    async def send_text(self, text: str) -> None:
        """Send a text input into the realtime session."""
        raise NotImplementedError

    @abc.abstractmethod
    async def send_audio_pcm(self, pcm: PcmData, *, mime_rate: int = 48000) -> None:
        """Send audio PCM into the realtime session (provider should resample if needed)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def send_image(
        self,
        *,
        data: Optional[bytes] = None,
        url: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> None:
        """Send an image (bytes or URL) into the realtime session."""
        raise NotImplementedError

    @abc.abstractmethod
    async def send_video_frame(
        self,
        *,
        data: bytes,
        mime_type: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Send a single encoded video frame (e.g. PNG/JPEG bytes) into the session."""
        raise NotImplementedError

    def emit_transcript(self, text: str, *, is_final: bool = False) -> None:
        """Emit a standardized transcript delta event to integrate with chat flow."""
        event = LLMStreamDeltaEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            delta=None,
            normalized_delta={"type": "text", "text": text},
            event_name="transcript.delta",
            is_final=is_final,
        )
        register_global_event(event)
        self.emit("delta", event)

    async def close(self):
        # If connected, disconnect first
        if self._is_connected:
            await self.disconnect("service_closed")
        await super().close()


class Conversation:
    """Lightweight helper to accumulate messages for turn-based interactions.

    This is purely optional and provider-agnostic; it helps build a message list to
    pass into LLM.generate/stream.
    """

    def __init__(self, system: Optional[str] = None):
        self.messages: List[Message] = []
        if system:
            self.add_system(system)

    def add_system(self, text: str) -> None:
        self.messages.append(
            {
                "role": Role.SYSTEM,
                "content": [{"type": "text", "text": text}],
            }
        )

    def add_user_text(self, text: str) -> None:
        self.messages.append(
            {
                "role": Role.USER,
                "content": [{"type": "text", "text": text}],
            }
        )

    def add_user_image(
        self,
        *,
        data: Optional[bytes] = None,
        url: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> None:
        if (data is None) == (url is None):
            raise ValueError("Provide exactly one of data or url for image input")

        image_part: Union[ImageBytesPart, ImageURLPart]
        if data is not None:
            image_part = {"type": "image", "data": data}
        else:
            image_part = {"type": "image", "url": url}  # type: ignore[arg-type]

        if mime_type is not None:
            image_part["mime_type"] = mime_type

        self.messages.append(
            {
                "role": Role.USER,
                "content": [image_part],
            }
        )

    def add_assistant_text(self, text: str) -> None:
        self.messages.append(
            {
                "role": Role.ASSISTANT,
                "content": [{"type": "text", "text": text}],
            }
        )

    def to_messages(self) -> List[Message]:
        return list(self.messages)
