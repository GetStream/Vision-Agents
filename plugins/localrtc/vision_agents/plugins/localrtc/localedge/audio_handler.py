"""Audio processing and streaming logic for LocalEdge.

This module handles audio input/output track creation, audio capture streaming,
and subscriber management for audio data.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData as StreamPcmData
from vision_agents.core.edge import events
from vision_agents.core.edge.types import OutputAudioTrack, Participant
from vision_agents.core.events import EventManager
from vision_agents.core.types import PcmData, TrackType

if TYPE_CHECKING:
    from .config import LocalEdgeConfig

logger = logging.getLogger(__name__)

# Constant user ID for local audio (microphone input from human user)
LOCAL_USER_ID = "local-user"


class AudioHandler:
    """Handles audio input/output tracks and streaming for LocalEdge.

    This class manages audio device access, track creation, and continuous
    audio capture streaming with support for multiple subscribers.

    Attributes:
        audio_device: Audio input device identifier
        speaker_device: Audio output device identifier
        sample_rate: Audio input sampling rate in Hz
        channels: Number of audio channels
        custom_pipeline: Optional GStreamer pipeline configuration
        events: Event manager for emitting audio events
    """

    def __init__(
        self,
        audio_device: Union[str, int],
        speaker_device: Union[str, int],
        sample_rate: int,
        channels: int,
        custom_pipeline: Optional[Dict[str, Any]],
        events: EventManager,
        config: Optional["LocalEdgeConfig"] = None,
    ) -> None:
        """Initialize the audio handler.

        Args:
            audio_device: Audio input device identifier
            speaker_device: Audio output device identifier
            sample_rate: Audio sampling rate in Hz
            channels: Number of audio channels
            custom_pipeline: Optional GStreamer pipeline configuration
            events: Event manager for emitting audio events
            config: Optional LocalEdgeConfig for audio settings
        """
        from .config import LocalEdgeConfig

        self.config = config if config is not None else LocalEdgeConfig()
        self.audio_device = audio_device
        self.speaker_device = speaker_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.custom_pipeline = custom_pipeline
        self.events = events

        # Track references
        self._audio_input_track: Optional[Any] = None
        self._audio_output_track: Optional[Any] = None

        # Subscriber management
        self._track_subscribers: Dict[str, List[Callable[[PcmData], None]]] = {}

        # Audio capture streaming
        self._audio_capture_thread: Optional[threading.Thread] = None
        self._audio_capture_running: bool = False

    def create_audio_input_track(self) -> Any:
        """Create and start an audio input track for microphone capture.

        Returns:
            AudioInputTrack or GStreamerAudioInputTrack instance
        """
        from ..tracks import AudioInputTrack, GStreamerAudioInputTrack

        if self._audio_input_track is None and self.audio_device is not None:
            logger.info(
                f"[LOCALRTC] Creating AudioInputTrack: device={self.audio_device}, "
                f"sample_rate={self.sample_rate}, channels={self.channels}"
            )

            if self.custom_pipeline and "audio_source" in self.custom_pipeline:
                # Use GStreamer pipeline
                self._audio_input_track = GStreamerAudioInputTrack(
                    pipeline=self.custom_pipeline["audio_source"],
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )
            else:
                # Use default device access
                self._audio_input_track = AudioInputTrack(
                    device=self.audio_device,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )

            # Start the persistent audio input stream
            self._audio_input_track.start()
            logger.info("[LOCALRTC] Audio input track started")

        return self._audio_input_track

    def create_audio_output_track(
        self,
        output_sample_rate: int,
        output_channels: int,
    ) -> OutputAudioTrack:
        """Create an audio output track for speaker playback.

        Args:
            output_sample_rate: Sample rate for output audio in Hz
            output_channels: Number of output channels (1=mono, 2=stereo)

        Returns:
            OutputAudioTrack instance for audio streaming
        """
        from ..tracks import AudioOutputTrack, GStreamerAudioOutputTrack

        if self._audio_output_track is None:
            logger.info(
                f"[LOCALRTC] Creating AudioOutputTrack: "
                f"sample_rate={output_sample_rate}, channels={output_channels}"
            )

            if self.custom_pipeline and "audio_sink" in self.custom_pipeline:
                # Use GStreamer pipeline
                self._audio_output_track = GStreamerAudioOutputTrack(
                    pipeline=self.custom_pipeline["audio_sink"],
                    sample_rate=output_sample_rate,
                    channels=output_channels,
                )
            else:
                # Use default device access
                self._audio_output_track = AudioOutputTrack(
                    device=self.speaker_device,
                    sample_rate=output_sample_rate,
                    channels=output_channels,
                )

        return self._audio_output_track

    def get_audio_input_track(self) -> Optional[Any]:
        """Get the current audio input track.

        Returns:
            Current audio input track or None if not created
        """
        return self._audio_input_track

    def get_audio_output_track(self) -> Optional[OutputAudioTrack]:
        """Get the current audio output track.

        Returns:
            Current audio output track or None if not created
        """
        return self._audio_output_track

    def add_track_subscriber(
        self, track_id: str, callback: Callable[[PcmData], None]
    ) -> None:
        """Add a subscriber callback for a track.

        Args:
            track_id: The ID of the track to subscribe to
            callback: Callback function to handle PCM data
        """
        # Initialize subscriber list for this track_id if not exists
        if track_id not in self._track_subscribers:
            self._track_subscribers[track_id] = []

        # Add callback to subscribers list
        self._track_subscribers[track_id].append(callback)

        # Start streaming for audio tracks
        if not self._audio_capture_running:
            self.start_audio_capture_stream()

    def start_audio_capture_stream(self) -> None:
        """Start continuous audio capture stream for subscribed callbacks."""
        if self._audio_capture_running or self._audio_input_track is None:
            return

        self._audio_capture_running = True
        self._audio_capture_thread = threading.Thread(
            target=self._audio_capture_loop, daemon=True
        )
        self._audio_capture_thread.start()
        logger.info("[LOCALRTC] Audio capture stream started")

    def stop_audio_capture_stream(self) -> None:
        """Stop continuous audio capture stream."""
        if not self._audio_capture_running:
            return

        self._audio_capture_running = False

        if (
            self._audio_capture_thread is not None
            and self._audio_capture_thread.is_alive()
        ):
            self._audio_capture_thread.join(timeout=self.config.audio.thread_join_timeout)
            self._audio_capture_thread = None

        logger.info("[LOCALRTC] Audio capture stream stopped")

    def stop_all_tracks(self) -> None:
        """Stop and cleanup all audio tracks."""
        # Stop audio capture streaming
        self.stop_audio_capture_stream()

        # Stop and cleanup audio output track
        if self._audio_output_track is not None:
            self._audio_output_track.stop()
            self._audio_output_track = None

        # Stop and cleanup audio input track
        if self._audio_input_track is not None:
            self._audio_input_track.stop()
            self._audio_input_track = None

        # Clear subscribers
        self._track_subscribers.clear()

    def _audio_capture_loop(self) -> None:
        """Continuous audio capture loop that invokes subscriber callbacks.

        This runs in a background thread and continuously captures audio chunks,
        then invokes all registered callbacks with PcmData and emits events.
        """
        if self._audio_input_track is None:
            return

        # Capture audio chunks using configured duration
        chunk_duration = self.config.audio.capture_chunk_duration

        # Create a participant for local audio (represents the human user)
        local_participant = Participant(original=None, user_id=LOCAL_USER_ID)

        while self._audio_capture_running:
            try:
                # Capture audio chunk (returns core PcmData with bytes)
                pcm_data = self._audio_input_track.capture(duration=chunk_duration)

                # Convert core PcmData (bytes) to StreamPcmData (numpy) for AudioQueue compatibility
                # The AudioQueue and Agent pipeline expect GetStream's PcmData format
                audio_samples = np.frombuffer(pcm_data.data, dtype=np.int16)
                stream_pcm = StreamPcmData(
                    sample_rate=pcm_data.sample_rate,
                    format=AudioFormat.S16,
                    samples=audio_samples,
                    channels=pcm_data.channels,
                    participant=local_participant,
                )

                # Emit AudioReceivedEvent for Agent subscription
                # This is the primary mechanism for delivering audio to the Agent
                logger.debug(
                    f"[LOCALRTC] Captured {len(pcm_data.data)} bytes, emitting AudioReceivedEvent"
                )
                self.events.send(
                    events.AudioReceivedEvent(
                        plugin_name="localrtc",
                        pcm_data=stream_pcm,
                        participant=local_participant,
                    )
                )

                # Invoke all audio subscribers (legacy callback mechanism)
                audio_subscribers = self._track_subscribers.get("audio", [])
                audio_subscribers.extend(
                    self._track_subscribers.get(str(TrackType.AUDIO), [])
                )

                for callback in audio_subscribers:
                    try:
                        callback(pcm_data)
                    except Exception as e:
                        # Log error but continue processing other callbacks
                        logger.error(f"Error in audio track subscriber callback: {e}")

            except Exception as e:
                if self._audio_capture_running:
                    # Only log if we're still supposed to be running
                    logger.error(f"Error in audio capture loop: {e}")
                    import time

                    time.sleep(self.config.audio.error_retry_delay)
                else:
                    # Shutting down, exit gracefully
                    break
