"""Local RTC Edge Transport core implementation."""

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import aiortc
from vision_agents.core.edge import events
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.types import OutputAudioTrack, User
from vision_agents.core.events import EventManager
from vision_agents.core.protocols import Room
from vision_agents.core.types import PcmData, TrackType

from .audio_handler import AudioHandler
from .format_negotiation import AudioFormatNegotiator
from .video_handler import VideoHandler

logger = logging.getLogger(__name__)

# Try to import GStreamer
try:
    import gi

    gi.require_version("Gst", "1.0")
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
            sample_rate: Audio input sampling rate in Hz (default: 16000)
                Note: Audio output is automatically set to 24000 Hz to match
                Gemini Realtime API's native format and avoid resampling issues.
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
            >>> edge = LocalEdge(audio_device="default", video_device=0)
            >>>
            >>> # Using custom GStreamer pipelines
            >>> pipeline = {
            ...     "audio_source": "alsasrc device=hw:0 ! audioconvert ! audioresample",
            ...     "video_source": "v4l2src device=/dev/video0 ! videoconvert",
            ...     "audio_sink": "alsasink device=hw:0"
            ... }
            >>> edge = LocalEdge(custom_pipeline=pipeline)
        """
        super().__init__()

        # Initialize event manager for edge transport events
        self.events = EventManager()
        # Register edge events so subscribers can receive them
        self.events.register_events_from_module(events)

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

        # Initialize handlers
        self._audio_handler = AudioHandler(
            audio_device=audio_device,
            speaker_device=speaker_device,
            sample_rate=sample_rate,
            channels=channels,
            custom_pipeline=custom_pipeline,
            events=self.events,
        )

        self._video_handler = VideoHandler(
            video_device=video_device,
            custom_pipeline=custom_pipeline,
        )

        self._format_negotiator = AudioFormatNegotiator(
            input_sample_rate=sample_rate,
            input_channels=channels,
        )

        # Track state
        self._user: Optional[User] = None
        self._rooms: Dict[str, Any] = {}

    @property
    def _negotiated_output_sample_rate(self) -> Optional[int]:
        """Get negotiated output sample rate (backward compatibility)."""
        return self._format_negotiator.output_sample_rate

    @property
    def _negotiated_output_channels(self) -> Optional[int]:
        """Get negotiated output channels (backward compatibility)."""
        return self._format_negotiator.output_channels

    @staticmethod
    def list_devices() -> Dict[str, List[Dict[str, Any]]]:
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
            >>> devices = LocalEdge.list_devices()
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
        from ..devices import list_audio_inputs, list_audio_outputs, list_video_inputs

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

    def create_audio_track(
        self, framerate: int = 48000, stereo: bool = True, **kwargs: Any
    ) -> OutputAudioTrack:
        """Create an output audio track.

        Uses GStreamer if custom_pipeline is configured with audio_sink,
        otherwise uses default device access.

        Note: For LocalRTC, we use 24000Hz mono to match Gemini's native output
        format and avoid resampling issues. The framerate/stereo parameters
        are accepted for API compatibility but may be adjusted for local playback.

        Args:
            framerate: Audio sample rate in Hz (default: 48000 for WebRTC compatibility)
            stereo: Whether to use stereo (2 channels) or mono (1 channel)
            **kwargs: Additional parameters for future compatibility.

        Returns:
            An OutputAudioTrack instance for audio streaming.
        """
        # Get negotiated format or use defaults
        output_sample_rate, output_channels = self._format_negotiator.get_output_format()

        return self._audio_handler.create_audio_output_track(
            output_sample_rate=output_sample_rate,
            output_channels=output_channels,
        )

    async def close(self) -> None:
        """Close the edge transport and clean up resources."""
        # Stop all audio/video tracks
        self._audio_handler.stop_all_tracks()
        self._video_handler.stop_all_tracks()

        # Leave all rooms
        for room in list(self._rooms.values()):
            await room.leave()
        self._rooms.clear()

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

    async def create_call(self, call_type: str, call_id: str) -> Room:
        """Create a call/room with the given type and ID.

        For LocalEdge, this creates a LocalRoom instance that manages
        local audio/video streams without external RTC infrastructure.

        Args:
            call_type: The type of call (used as room_type)
            call_id: Unique identifier for the call (used as room_id)

        Returns:
            A LocalRoom instance representing the created call
        """
        from ..room import LocalRoom

        # Create or get existing room
        if call_id not in self._rooms:
            self._rooms[call_id] = LocalRoom(room_id=call_id, room_type=call_type)

        return self._rooms[call_id]

    async def join(self, *args: Any, **kwargs: Any) -> Room:
        """Join a room and start device capture.

        This method supports two calling conventions:

        **Recommended (with agent for audio negotiation):**
            >>> room = await edge.join(agent, room_id="call-1", room_type="default")

        **Legacy (without agent):**
            >>> room = await edge.join(room_id="call-1", room_type="default")

        Args:
            *args: Positional arguments for join configuration.
                First positional argument should be the Agent instance for audio
                format negotiation (recommended). When an agent is provided, the
                output audio format is automatically negotiated based on the LLM
                provider's requirements.
            **kwargs: Keyword arguments for join configuration.
                - room_id (str): Unique identifier for the room. Defaults to "local-room".
                - room_type (str): Type of room. Defaults to "default".

        Returns:
            A Room instance representing the joined room.

        Note:
            When an agent is not provided, the output audio format defaults to
            24000Hz mono (Gemini's native format). For best results, always pass
            the agent instance to enable automatic audio format negotiation.

        Example:
            >>> edge = LocalEdge(audio_device="default")
            >>> agent = Agent(edge=edge, llm=gemini.Realtime())
            >>> call = await agent.create_call("default", "my-call")
            >>> room = await edge.join(agent, room_id="my-call")
        """
        from ..room import LocalRoom

        # Extract agent from args if provided (new Agent API passes agent as first arg)
        agent = args[0] if args and hasattr(args[0], "llm") else None

        # Negotiate audio format with LLM provider if agent is available
        if agent is not None:
            self._format_negotiator.negotiate_format(agent)
            logger.info("[AUDIO NEGOTIATION] Format negotiated with agent")
        else:
            # Set default values when no agent is provided
            # Create a mock agent without LLM to trigger default negotiation
            class _MockAgent:
                pass

            self._format_negotiator.negotiate_format(_MockAgent())
            logger.info(
                "[AUDIO NEGOTIATION] No agent provided, using default output format"
            )

        room_id = kwargs.get("room_id", "local-room")
        room_type = kwargs.get("room_type", "default")

        # Create or get existing room
        if room_id not in self._rooms:
            self._rooms[room_id] = LocalRoom(room_id=room_id, room_type=room_type)

        # Create and start audio input track for microphone capture
        # This is necessary for LocalRTC because there's no external RTC infrastructure
        # pushing audio to us - we need to capture it locally
        self._audio_handler.create_audio_input_track()
        # Start the capture loop to emit AudioReceivedEvents
        self._audio_handler.start_audio_capture_stream()
        logger.info("[LOCALRTC] Audio capture started")

        return self._rooms[room_id]

    async def publish_tracks(
        self, room: Any, audio_track: Any = None, video_track: Any = None
    ) -> None:
        """Publish audio and video tracks to the room.

        **Recommended calling convention:**
            >>> await edge.publish_tracks(room, audio_track=audio, video_track=video)
            >>> await edge.publish_tracks(room, audio_track=audio)  # Audio only

        **Legacy calling convention (deprecated):**
            >>> await edge.publish_tracks(audio, video)  # Will be removed in v2.0

        Args:
            room: The room to publish tracks to. In the recommended convention, this
                should be a Room instance. In the legacy convention, this is the
                audio_track.
            audio_track: Audio track to publish (e.g., AudioInputTrack instance).
                Optional - only required if you want to publish audio.
            video_track: Video track to publish (e.g., VideoInputTrack instance).
                Optional - only required if you want to publish video.

        Note:
            The legacy calling convention without explicit room parameter is
            deprecated and will be removed in version 2.0. Please update your
            code to use the new convention with explicit room parameter.

        Example:
            >>> edge = LocalEdge()
            >>> room = await edge.join(agent, room_id="my-call")
            >>> audio = AudioInputTrack(device="default")
            >>> video = VideoInputTrack(device=0)
            >>> await edge.publish_tracks(room, audio_track=audio, video_track=video)
        """
        from ..room import LocalRoom
        from ..tracks import AudioInputTrack, VideoInputTrack

        # Detect legacy calling convention and issue deprecation warning
        is_legacy = False
        if audio_track is None and video_track is None:
            # Legacy: publish_tracks(audio_track)
            is_legacy = True
        elif video_track is None and not isinstance(room, (LocalRoom,)):
            # Legacy: publish_tracks(audio_track, video_track)
            # Check if room looks like a track instead of a Room
            if hasattr(room, "capture") or hasattr(room, "start"):
                is_legacy = True

        if is_legacy:
            warnings.warn(
                "Calling publish_tracks without explicit room parameter is deprecated "
                "and will be removed in version 2.0. Use: "
                "await edge.publish_tracks(room, audio_track=audio, video_track=video)",
                DeprecationWarning,
                stacklevel=2,
            )

        # Handle both calling conventions
        if audio_track is None and video_track is None:
            # Legacy calling convention: publish_tracks(audio, video)
            # room parameter actually contains audio_track
            actual_audio = room
            actual_video = None
            actual_room = None
        elif video_track is None and is_legacy:
            # Legacy calling convention: publish_tracks(audio, video)
            # room contains audio, audio_track contains video
            actual_audio = room
            actual_video = audio_track
            actual_room = None
        else:
            # New calling convention: publish_tracks(room, audio, video)
            actual_audio = audio_track
            actual_video = video_track
            actual_room = room

        # Store references to the tracks
        if actual_audio is not None:
            if isinstance(actual_audio, AudioInputTrack):
                # Start the audio capture loop to emit AudioReceivedEvents
                # This is necessary because the Agent subscribes to events, not callbacks
                self._audio_handler.start_audio_capture_stream()
                logger.info("[LOCALRTC] Audio capture started via publish_tracks")
            # Store in room if provided
            if isinstance(actual_room, LocalRoom):
                actual_room._tracks[TrackType.AUDIO] = actual_audio

        if actual_video is not None:
            if isinstance(actual_video, VideoInputTrack):
                pass  # Video track reference is stored in video handler
            # Store in room if provided
            if isinstance(actual_room, LocalRoom):
                actual_room._tracks[TrackType.VIDEO] = actual_video

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
        # Determine track type from track_id
        track_type = self._get_track_type_from_id(track_id)

        # Delegate to audio handler for audio tracks
        if track_type == TrackType.AUDIO:
            self._audio_handler.add_track_subscriber(track_id, callback)

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
        from ..room import LocalRoom

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
