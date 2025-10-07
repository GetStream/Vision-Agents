import abc
import logging
import inspect
import time
import uuid
from typing import Optional, Dict, Any, Union, Iterator, AsyncIterator

from getstream.video.rtc.audio_track import AudioStreamTrack
from vision_agents.core.events.manager import EventManager

from . import events
from .events import (
    TTSAudioEvent,
    TTSSynthesisStartEvent,
    TTSSynthesisCompleteEvent,
    TTSErrorEvent,
)
from vision_agents.core.events import PluginInitializedEvent, PluginClosedEvent

logger = logging.getLogger(__name__)


class TTS(abc.ABC):
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
        self.events = EventManager()
        self.events.register_events_from_module(events, ignore_not_compatible=True)
        self.events.send(PluginInitializedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="TTS",
            provider=self.provider_name,
        ))

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

    def get_required_framerate(self) -> int:
        """
        Get the required framerate for the audio track.
        
        This method should be overridden by subclasses to return their specific
        framerate requirement. Defaults to 16000 Hz.
        
        Returns:
            The required framerate in Hz
        """
        return 16000

    def get_required_stereo(self) -> bool:
        """
        Get whether the audio track should be stereo or mono.
        
        This method should be overridden by subclasses to return their specific
        stereo requirement. Defaults to False (mono).
        
        Returns:
            True if stereo is required, False for mono
        """
        return False

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

            self.events.send(TTSSynthesisStartEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                text=text,
                synthesis_id=synthesis_id,
                user_metadata=user,
            ))

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

                audio_event = TTSAudioEvent(
                    session_id=self.session_id,
                    plugin_name=self.provider_name,
                    audio_data=audio_data,
                    synthesis_id=synthesis_id,
                    text_source=text,
                    user_metadata=user,
                    sample_rate=self._track.framerate if self._track else 16000,
                )
                self.events.send(audio_event)  # Structured event
            elif inspect.isasyncgen(audio_data):
                async for chunk in audio_data:
                    if isinstance(chunk, bytes):
                        total_audio_bytes += len(chunk)
                        audio_chunks += 1
                        await self._track.write(chunk)

                        # Emit structured audio event
                        self.events.send(TTSAudioEvent(
                            session_id=self.session_id,
                            plugin_name=self.provider_name,
                            audio_data=chunk,
                            synthesis_id=synthesis_id,
                            text_source=text,
                            user_metadata=user,
                            chunk_index=audio_chunks - 1,
                            is_final_chunk=False,  # We don't know if it's final yet
                            sample_rate=self._track.framerate if self._track else 16000,
                        ))
                    else:  # assume it's a Cartesia TTS chunk object
                        total_audio_bytes += len(chunk.data)
                        audio_chunks += 1
                        await self._track.write(chunk.data)

                        self.events.send(TTSAudioEvent(
                            session_id=self.session_id,
                            plugin_name=self.provider_name,
                            audio_data=chunk.data,
                            synthesis_id=synthesis_id,
                            text_source=text,
                            user_metadata=user,
                            chunk_index=audio_chunks - 1,
                            is_final_chunk=False,  # We don't know if it's final yet
                            sample_rate=self._track.framerate if self._track else 16000,
                        ))
            elif hasattr(audio_data, "__iter__") and not isinstance(
                audio_data, (str, bytes, bytearray)
            ):
                for chunk in audio_data:
                    total_audio_bytes += len(chunk)
                    audio_chunks += 1
                    await self._track.write(chunk)

                    self.events.send(TTSAudioEvent(
                        session_id=self.session_id,
                        plugin_name=self.provider_name,
                        audio_data=chunk,
                        synthesis_id=synthesis_id,
                        text_source=text,
                        user_metadata=user,
                        chunk_index=audio_chunks - 1,
                        is_final_chunk=False,  # We don't know if it's final yet
                        sample_rate=self._track.framerate if self._track else 16000,
                    ))
            else:
                raise TypeError(
                    f"Unsupported return type from synthesize: {type(audio_data)}"
                )

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

            self.events.send(TTSSynthesisCompleteEvent(
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
            ))
        except Exception as e:
            self.events.send(TTSErrorEvent(
                session_id=self.session_id,
                plugin_name=self.provider_name,
                error=e,
                context="synthesis",
                text_source=text,
                synthesis_id=synthesis_id,
                user_metadata=user,
            ))
            # ASK: why ?
            # Re-raise to allow the caller to handle the error
            raise

    async def close(self):
        """Close the TTS service and release any resources."""
        self.events.send(PluginClosedEvent(
            session_id=self.session_id,
            plugin_name=self.provider_name,
            plugin_type="TTS",
            provider=self.provider_name,
            cleanup_successful=True,
        ))
