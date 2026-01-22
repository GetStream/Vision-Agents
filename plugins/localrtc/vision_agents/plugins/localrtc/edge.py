"""Local RTC Edge Transport implementation."""

from typing import Any, Callable, Dict, Optional, Union

import aiortc
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.types import OutputAudioTrack, User
from vision_agents.core.protocols import Room
from vision_agents.core.types import PcmData, TrackType

from .room import LocalRoom
from .tracks import AudioInputTrack, AudioOutputTrack, VideoInputTrack


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
    """

    def __init__(
        self,
        audio_device: Union[str, int] = "default",
        video_device: Union[int, str] = 0,
        speaker_device: Union[str, int] = "default",
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        """Initialize the local edge transport.

        Args:
            audio_device: Audio input device identifier (default: "default")
            video_device: Video input device identifier (default: 0)
            speaker_device: Audio output device identifier (default: "default")
            sample_rate: Audio sampling rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
        """
        super().__init__()
        self.audio_device = audio_device
        self.video_device = video_device
        self.speaker_device = speaker_device
        self.sample_rate = sample_rate
        self.channels = channels

        # Track state
        self._user: Optional[User] = None
        self._audio_input_track: Optional[AudioInputTrack] = None
        self._audio_output_track: Optional[AudioOutputTrack] = None
        self._video_input_track: Optional[VideoInputTrack] = None
        self._rooms: Dict[str, LocalRoom] = {}
        self._track_subscribers: Dict[str, Callable[[PcmData], None]] = {}

    async def create_user(self, user: User) -> None:
        """Create a user in the local edge transport.

        Args:
            user: User object containing user information.
        """
        self._user = user

    def create_audio_track(self) -> OutputAudioTrack:
        """Create an output audio track.

        Returns:
            An OutputAudioTrack instance for audio streaming.
        """
        if self._audio_output_track is None:
            self._audio_output_track = AudioOutputTrack(
                device=self.speaker_device,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
        return self._audio_output_track

    async def close(self) -> None:
        """Close the edge transport and clean up resources."""
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

        Args:
            track_id: The ID of the track to subscribe to.
            callback: Callback function to handle PCM data.

        Returns:
            A MediaStreamTrack if available, None otherwise.
        """
        self._track_subscribers[track_id] = callback
        # For local edge, we don't use aiortc MediaStreamTracks
        return None
