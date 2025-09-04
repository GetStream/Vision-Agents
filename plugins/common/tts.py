import abc
import logging
import inspect
import time
import uuid
from typing import Optional, Dict, Any, Union, Iterator, AsyncIterator

from pyee.asyncio import AsyncIOEventEmitter
from getstream.video.rtc.audio_track import AudioStreamTrack

from .events import (
    TTSAudioEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
    TTSErrorEvent,
    PluginInitializedEvent,
    PluginClosedEvent,
)
from .event_utils import register_global_event

logger = logging.getLogger(__name__)


class TTS(AsyncIOEventEmitter, abc.ABC):
    """
    Text-to-Speech base class.

    This abstract class provides the interface for text-to-speech implementations.
    It handles:
    - Converting text to speech
    - Sending audio data to an output track
    - Emitting audio events

    Events:
        - audio: Emitted when an audio chunk is available.
            Args: audio_data (bytes), user_metadata (dict)
        - error: Emitted when an error occurs during speech synthesis.
            Args: error (Exception)

    Implementations should inherit from this class and implement the synthesize method.
    """

    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the TTS base class.

        Args:
            provider_name: Name of the TTS provider (e.g., "cartesia", "elevenlabs")
        """
        super().__init__()
        self._track: Optional[AudioStreamTrack] = None
        self.session_id = str(uuid.uuid4())
        self.provider_name = provider_name or self.__class__.__name__

        logger.debug(
            "Initialized TTS base class",
            extra={
                "session_id": self.session_id,
                "provider": self.provider_name,
            },
        )

        # Emit initialization event
        init_event = PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="TTS",
            provider=self.provider_name,
        )
        register_global_event(init_event)
        self.emit("initialized", init_event)

    def set_output_track(self, track: AudioStreamTrack) -> None:
        """
        Set the audio track to output speech to.

        Args:
            track: The audio track object that will receive speech audio
        """
        self._track = track

    @property
    def track(self):
        """Get the current output track."""
        return self._track

    @abc.abstractmethod
    async def stream_audio(
        self, text: str, *args, **kwargs
    ) -> Union[bytes, Iterator[bytes], AsyncIterator[bytes]]:
        """
        Convert text to speech audio data.

        This method must be implemented by subclasses.

        Args:
            text: The text to convert to speech
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Audio data as bytes, an iterator of audio chunks, or an async iterator of audio chunks
        """
        pass

    @abc.abstractmethod
    async def stop_audio(self) -> None:
        """
        Clears the queue and stops playing audio.
        This method can be used manually or under the hood in response to turn events.

        This method must be implemented by subclasses.


        Returns:
            None
        """
        pass

    async def send(
        self, text: str, user: Optional[Dict[str, Any]] = None, *args, **kwargs
    ):
        """
        Convert text to speech, send to the output track, and emit an audio event.

        Args:
            text: The text to convert to speech
            user: Optional user metadata to include with the audio event
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Raises:
            ValueError: If no output track has been set
        """
        if self._track is None:
            raise ValueError("No output track set. Call set_output_track() first.")

        try:
            # Log start of synthesis
            start_time = time.time()
            synthesis_id = str(uuid.uuid4())

            logger.debug(
                "Starting text-to-speech synthesis", extra={"text_length": len(text)}
            )

            # Emit synthesis start event
            start_event = TTSSynthesisStartEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                synthesis_id=synthesis_id,
                user_metadata=user,
            )
            register_global_event(start_event)
            self.emit("synthesis_start", start_event)

            # Synthesize audio
            audio_data = await self.stream_audio(text, *args, **kwargs)

            # Calculate synthesis time
            synthesis_time = time.time() - start_time

            # Track total audio duration and bytes
            total_audio_bytes = 0
            audio_chunks = 0

            if isinstance(audio_data, bytes):
                total_audio_bytes = len(audio_data)
                audio_chunks = 1
                await self._track.write(audio_data)

                # Emit structured audio event
                audio_event = TTSAudioEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    audio_data=audio_data,
                    synthesis_id=synthesis_id,
                    text_source=text,
                    user_metadata=user,
                    sample_rate=self._track.framerate if self._track else 16000,
                )
                register_global_event(audio_event)
                self.emit("audio", audio_event)  # Structured event
            elif inspect.isasyncgen(audio_data):
                async for chunk in audio_data:
                    if isinstance(chunk, bytes):
                        total_audio_bytes += len(chunk)
                        audio_chunks += 1
                        await self._track.write(chunk)

                        # Emit structured audio event
                        audio_event = TTSAudioEvent(
                            session_id=self.session_id,
                            plugin_name=self.provider_name,
                            audio_data=chunk,
                            synthesis_id=synthesis_id,
                            text_source=text,
                            user_metadata=user,
                            chunk_index=audio_chunks - 1,
                            is_final_chunk=False,  # We don't know if it's final yet
                            sample_rate=self._track.framerate if self._track else 16000,
                        )
                        register_global_event(audio_event)
                        self.emit("audio", audio_event)  # Structured event
                    else:  # assume it's a Cartesia TTS chunk object
                        total_audio_bytes += len(chunk.data)
                        audio_chunks += 1
                        await self._track.write(chunk.data)

                        # Emit structured audio event
                        audio_event = TTSAudioEvent(
                            session_id=self.session_id,
                            plugin_name=self.provider_name,
                            audio_data=chunk.data,
                            synthesis_id=synthesis_id,
                            text_source=text,
                            user_metadata=user,
                            chunk_index=audio_chunks - 1,
                            is_final_chunk=False,  # We don't know if it's final yet
                            sample_rate=self._track.framerate if self._track else 16000,
                        )
                        register_global_event(audio_event)
                        self.emit("audio", audio_event)  # Structured event
            elif hasattr(audio_data, "__iter__") and not isinstance(
                audio_data, (str, bytes, bytearray)
            ):
                for chunk in audio_data:
                    total_audio_bytes += len(chunk)
                    audio_chunks += 1
                    await self._track.write(chunk)

                    # Emit structured audio event
                    audio_event = TTSAudioEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        audio_data=chunk,
                        synthesis_id=synthesis_id,
                        text_source=text,
                        user_metadata=user,
                        chunk_index=audio_chunks - 1,
                        is_final_chunk=False,  # We don't know if it's final yet
                        sample_rate=self._track.framerate if self._track else 16000,
                    )
                    register_global_event(audio_event)
                    self.emit("audio", audio_event)  # Structured event
            else:
                raise TypeError(
                    f"Unsupported return type from synthesize: {type(audio_data)}"
                )

            # Log completion with timing information
            end_time = time.time()
            total_time = end_time - start_time

            # Estimate audio duration - this is approximate without knowing format details
            # Use track framerate if available, otherwise assume 16kHz
            sample_rate = self._track.framerate if self._track else 16000
            # For s16 format (16-bit samples), each byte is half a sample
            estimated_audio_duration_ms = (total_audio_bytes / 2) / (sample_rate / 1000)

            real_time_factor = (
                (synthesis_time * 1000) / estimated_audio_duration_ms
                if estimated_audio_duration_ms > 0
                else None
            )

            # Emit synthesis completion event
            completion_event = TTSSynthesisCompleteEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                synthesis_id=synthesis_id,
                text=text,
                user_metadata=user,
                total_audio_bytes=total_audio_bytes,
                synthesis_time_ms=synthesis_time * 1000,
                audio_duration_ms=estimated_audio_duration_ms,
                chunk_count=audio_chunks,
                real_time_factor=real_time_factor,
            )
            register_global_event(completion_event)
            self.emit("synthesis_complete", completion_event)

            logger.info(
                "Text-to-speech synthesis completed",
                extra={
                    "event_id": completion_event.event_id,
                    "text_length": len(text),
                    "synthesis_time_ms": synthesis_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "audio_bytes": total_audio_bytes,
                    "audio_chunks": audio_chunks,
                    "estimated_audio_duration_ms": estimated_audio_duration_ms,
                    "real_time_factor": real_time_factor,
                    "sample_rate": sample_rate,
                },
            )

        except Exception as e:
            # Emit structured error event
            error_event = TTSErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=e,
                context="synthesis",
                text_source=text,
                synthesis_id=synthesis_id,
                user_metadata=user,
            )
            register_global_event(error_event)
            self.emit("error", error_event)  # New structured event
            self.emit("error_legacy", e)  # Backward compatibility
            # Re-raise to allow the caller to handle the error
            raise

    async def close(self):
        """Close the TTS service and release any resources."""
        # Emit closure event
        close_event = PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="TTS",
            provider=self.provider_name,
            cleanup_successful=True,
        )
        register_global_event(close_event)
        self.emit("closed", close_event)
