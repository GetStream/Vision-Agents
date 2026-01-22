"""Local RTC Edge Transport implementation."""

import threading
from typing import Any, Callable, Dict, List, Optional, Union

import aiortc
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.types import OutputAudioTrack, User
from vision_agents.core.protocols import Room
from vision_agents.core.types import PcmData, TrackType

from .devices import DeviceInfo, list_audio_inputs, list_audio_outputs, list_video_inputs
from .room import LocalRoom
from .tracks import (
    AudioInputTrack,
    AudioOutputTrack,
    VideoInputTrack,
    GStreamerAudioInputTrack,
    GStreamerAudioOutputTrack,
    GStreamerVideoInputTrack,
)

# Try to import GStreamer
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    GST_AVAILABLE = True
except (ImportError, ValueError):
    GST_AVAILABLE = False
    Gst = None  # type: ignore


class LocalEdge(EdgeTransport):
    """Local RTC Edge Transport for managing local audio/video streams.

    This class implements the EdgeTransport interface for local device access,
    allowing audio and video capture/playback without external RTC infrastructure.

    Attributes:
        audio_device: Audio input device identifier
        video_device: Video input device identifier
        speaker_device: Audio output device identifier
        sample_rate: Audio sampling rate in Hz
        channels: Number of audio channels
        custom_pipeline: Optional GStreamer pipeline configuration
    """

    def __init__(
        self,
        audio_device: Union[str, int] = "default",
        video_device: Union[int, str] = 0,
        speaker_device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
        custom_pipeline: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the local edge transport.

        Args:
            audio_device: Audio input device identifier (default: "default")
            video_device: Video input device identifier (default: 0)
            speaker_device: Audio output device identifier (default: "default")
            sample_rate: Audio sampling rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            custom_pipeline: Optional GStreamer pipeline configuration. When provided,
                uses GStreamer instead of default device access. Dictionary with keys:
                - audio_source: GStreamer pipeline string for audio input
                  Example: "pulsesrc ! audioconvert ! audioresample"
                - video_source: GStreamer pipeline string for video input
                  Example: "v4l2src device=/dev/video0 ! videoconvert"
                - audio_sink: GStreamer pipeline string for audio output
                  Example: "autoaudiosink"
                Note: Requires PyGObject (gi) and GStreamer to be installed.

        Raises:
            RuntimeError: If custom_pipeline is provided but GStreamer is not available

        Example:
            >>> # Using default device access
            >>> edge = Edge(audio_device="default", video_device=0)
            >>>
            >>> # Using custom GStreamer pipelines
            >>> pipeline = {
            ...     "audio_source": "alsasrc device=hw:0 ! audioconvert ! audioresample",
            ...     "video_source": "v4l2src device=/dev/video0 ! videoconvert",
            ...     "audio_sink": "alsasink device=hw:0"
            ... }
            >>> edge = Edge(custom_pipeline=pipeline)
        """
        super().__init__()

        # Validate GStreamer availability if custom_pipeline is provided
        if custom_pipeline is not None and not GST_AVAILABLE:
            raise RuntimeError(
                "GStreamer is not available. To use custom pipelines, install PyGObject and GStreamer:\n"
                "  Ubuntu/Debian: sudo apt-get install python3-gi gstreamer1.0-tools gstreamer1.0-plugins-base\n"
                "  Fedora: sudo dnf install python3-gobject gstreamer1-tools gstreamer1-plugins-base\n"
                "  macOS: brew install pygobject3 gstreamer"
            )

        self.audio_device = audio_device
        self.video_device = video_device
        self.speaker_device = speaker_device
        self.sample_rate = sample_rate
        self.channels = channels
        self.custom_pipeline = custom_pipeline

        # Track state
        self._user: Optional[User] = None
        self._audio_input_track: Optional[Union[AudioInputTrack, GStreamerAudioInputTrack]] = None
        self._audio_output_track: Optional[Union[AudioOutputTrack, GStreamerAudioOutputTrack]] = None
        self._video_input_track: Optional[Union[VideoInputTrack, GStreamerVideoInputTrack]] = None
        self._rooms: Dict[str, LocalRoom] = {}
        self._track_subscribers: Dict[str, List[Callable[[PcmData], None]]] = {}
        self._audio_capture_thread: Optional[threading.Thread] = None
        self._audio_capture_running: bool = False

        # GStreamer pipeline state
        self._gst_audio_pipeline: Optional[Any] = None
        self._gst_video_pipeline: Optional[Any] = None
        self._gst_audio_sink_pipeline: Optional[Any] = None

    @staticmethod
    def list_devices() -> Dict[str, List[DeviceInfo]]:
        """List all available audio and video devices.

        This static method discovers and returns all available audio input,
        audio output, and video input devices on the system. It provides a
        convenient way to enumerate devices before creating an Edge instance.

        Returns:
            Dictionary with three keys:
                - audio_inputs: List of audio input devices
                - audio_outputs: List of audio output devices
                - video_inputs: List of video input devices
            Each device is a dict with 'name' and 'index' keys.

        Example:
            >>> devices = Edge.list_devices()
            >>> print("Audio inputs:")
            >>> for device in devices["audio_inputs"]:
            ...     print(f"  {device['index']}: {device['name']}")
            >>> print("Audio outputs:")
            >>> for device in devices["audio_outputs"]:
            ...     print(f"  {device['index']}: {device['name']}")
            >>> print("Video inputs:")
            >>> for device in devices["video_inputs"]:
            ...     print(f"  {device['index']}: {device['name']}")

        Note:
            Video device enumeration is not yet fully implemented and may
            return an empty list on some platforms.
        """
        return {
            "audio_inputs": list_audio_inputs(),
            "audio_outputs": list_audio_outputs(),
            "video_inputs": list_video_inputs(),
        }

    async def create_user(self, user: User) -> None:
        """Create a user in the local edge transport.

        Args:
            user: User object containing user information.
        """
        self._user = user

    def create_audio_track(self) -> OutputAudioTrack:
        """Create an output audio track.

        Uses GStreamer if custom_pipeline is configured with audio_sink,
        otherwise uses default device access.

        Returns:
            An OutputAudioTrack instance for audio streaming.
        """
        if self._audio_output_track is None:
            if self.custom_pipeline and "audio_sink" in self.custom_pipeline:
                # Use GStreamer pipeline
                self._audio_output_track = GStreamerAudioOutputTrack(
                    pipeline=self.custom_pipeline["audio_sink"],
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )
            else:
                # Use default device access
                self._audio_output_track = AudioOutputTrack(
                    device=self.speaker_device,
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                )
        return self._audio_output_track

    async def close(self) -> None:
        """Close the edge transport and clean up resources."""
        # Stop audio capture streaming
        self._stop_audio_capture_stream()

        # Stop and cleanup audio output track
        if self._audio_output_track is not None:
            self._audio_output_track.stop()
            self._audio_output_track = None

        # Stop and cleanup video input track
        if self._video_input_track is not None:
            self._video_input_track.stop()
            self._video_input_track = None

        # Cleanup audio input track
        self._audio_input_track = None

        # Leave all rooms
        for room in list(self._rooms.values()):
            await room.leave()
        self._rooms.clear()

        # Clear subscribers
        self._track_subscribers.clear()

        # Clear user
        self._user = None

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """Open a demo session.

        Args:
            *args: Positional arguments for demo configuration.
            **kwargs: Keyword arguments for demo configuration.
        """
        # For local edge, demo mode is a no-op since we're always in "demo" mode
        pass

    async def join(self, *args: Any, **kwargs: Any) -> Room:
        """Join a room.

        Args:
            *args: Positional arguments for join configuration.
            **kwargs: Keyword arguments for join configuration.

        Returns:
            A Room instance representing the joined room.
        """
        room_id = kwargs.get("room_id", "local-room")
        room_type = kwargs.get("room_type", "default")

        # Create or get existing room
        if room_id not in self._rooms:
            self._rooms[room_id] = LocalRoom(room_id=room_id, room_type=room_type)

        return self._rooms[room_id]

    async def publish_tracks(
        self, room: Room, audio_track: Any, video_track: Any
    ) -> None:
        """Publish audio and video tracks to the room.

        Args:
            room: The room to publish tracks to.
            audio_track: Audio track to publish.
            video_track: Video track to publish.
        """
        # Store references to the tracks
        if isinstance(room, LocalRoom):
            if audio_track is not None:
                if isinstance(audio_track, AudioInputTrack):
                    self._audio_input_track = audio_track
                room._tracks[TrackType.AUDIO] = audio_track

            if video_track is not None:
                if isinstance(video_track, VideoInputTrack):
                    self._video_input_track = video_track
                room._tracks[TrackType.VIDEO] = video_track

    async def create_conversation(
        self, call: Any, user: User, instructions: Any
    ) -> None:
        """Create a conversation in the call.

        Args:
            call: The call object.
            user: User object.
            instructions: Conversation instructions.
        """
        # For local edge, conversation creation is a no-op
        # This method is primarily used by external RTC services
        pass

    def add_track_subscriber(
        self, track_id: str, callback: Callable[[PcmData], None]
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        """Add a subscriber to a track.

        This method subscribes to audio or video tracks and invokes the callback
        with the appropriate data. For audio tracks, the callback receives PcmData.
        For video tracks, this method returns None as local RTC doesn't use
        aiortc MediaStreamTracks.

        Args:
            track_id: The ID of the track to subscribe to. Should match TrackType
                     values (e.g., "audio", "video") or room-specific track IDs.
            callback: Callback function to handle PCM data for audio tracks.

        Returns:
            A MediaStreamTrack if available (None for local edge as we don't use
            aiortc MediaStreamTracks).
        """
        # Initialize subscriber list for this track_id if not exists
        if track_id not in self._track_subscribers:
            self._track_subscribers[track_id] = []

        # Add callback to subscribers list
        self._track_subscribers[track_id].append(callback)

        # Determine track type from track_id
        track_type = self._get_track_type_from_id(track_id)

        # Start streaming for audio tracks
        if track_type == TrackType.AUDIO and not self._audio_capture_running:
            self._start_audio_capture_stream()

        # For video tracks, they use their own callback mechanism via start()
        # Video tracks are not managed through add_track_subscriber callbacks

        # For local edge, we don't use aiortc MediaStreamTracks
        return None

    def _get_track_type_from_id(self, track_id: str) -> Optional[TrackType]:
        """Determine track type from track ID.

        Args:
            track_id: The track identifier

        Returns:
            TrackType if determinable, None otherwise
        """
        # Check if track_id matches TrackType enum values
        track_id_lower = track_id.lower()
        if track_id_lower == "audio" or "audio" in track_id_lower:
            return TrackType.AUDIO
        elif track_id_lower == "video" or "video" in track_id_lower:
            return TrackType.VIDEO
        elif track_id_lower == "screenshare" or "screen" in track_id_lower:
            return TrackType.SCREENSHARE

        # Check if track exists in any room
        for room in self._rooms.values():
            if isinstance(room, LocalRoom):
                for track_type, track in room._tracks.items():
                    if track_id == str(track_type) or track_id in str(track_type):
                        return track_type

        return None

    def _start_audio_capture_stream(self) -> None:
        """Start continuous audio capture stream for subscribed callbacks."""
        if self._audio_capture_running or self._audio_input_track is None:
            return

        self._audio_capture_running = True
        self._audio_capture_thread = threading.Thread(
            target=self._audio_capture_loop, daemon=True
        )
        self._audio_capture_thread.start()

    def _stop_audio_capture_stream(self) -> None:
        """Stop continuous audio capture stream."""
        if not self._audio_capture_running:
            return

        self._audio_capture_running = False

        if self._audio_capture_thread is not None and self._audio_capture_thread.is_alive():
            self._audio_capture_thread.join(timeout=2.0)
            self._audio_capture_thread = None

    def _audio_capture_loop(self) -> None:
        """Continuous audio capture loop that invokes subscriber callbacks.

        This runs in a background thread and continuously captures audio chunks,
        then invokes all registered callbacks with PcmData.
        """
        if self._audio_input_track is None:
            return

        # Capture audio in 100ms chunks
        chunk_duration = 0.1  # seconds

        while self._audio_capture_running:
            try:
                # Capture audio chunk
                pcm_data = self._audio_input_track.capture(duration=chunk_duration)

                # Invoke all audio subscribers
                audio_subscribers = self._track_subscribers.get("audio", [])
                audio_subscribers.extend(self._track_subscribers.get(str(TrackType.AUDIO), []))

                for callback in audio_subscribers:
                    try:
                        callback(pcm_data)
                    except Exception as e:
                        # Log error but continue processing other callbacks
                        print(f"Error in audio track subscriber callback: {e}")

            except Exception as e:
                if self._audio_capture_running:
                    # Only log if we're still supposed to be running
                    print(f"Error in audio capture loop: {e}")
                    import time
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    # Shutting down, exit gracefully
                    break
