"""
Abstraction for stream vs other services here
"""

import abc

from typing import TYPE_CHECKING, Any, Callable, Optional

import aiortc
from pyee.asyncio import AsyncIOEventEmitter

from vision_agents.core.edge.types import User, OutputAudioTrack
from vision_agents.core.types import PcmData
from vision_agents.core.protocols import Room

if TYPE_CHECKING:
    pass


class EdgeTransport(AsyncIOEventEmitter, abc.ABC):
    """Abstract base class for edge transport layers.

    EdgeTransport defines the interface for real-time audio/video communication
    transport layers in Vision Agents. Concrete implementations include:
    - StreamEdge: GetStream-based WebRTC transport
    - LocalEdge: Local development transport using localrtc

    The transport layer is responsible for:
    - Managing WebRTC connections and tracks
    - Publishing and subscribing to audio/video streams
    - Converting between transport-specific and Vision Agents audio formats
    - Handling participant management and call lifecycle

    Audio Format Handling:
        Different transport implementations may use different default audio formats.
        All audio data is standardized using the PcmData type, which includes:
        - sample_rate: Samples per second (Hz)
        - channels: Number of audio channels (1=mono, 2=stereo)
        - bit_depth: Bits per sample (typically 16)
        - data: Raw PCM audio as numpy array
        - timestamp: Optional Unix timestamp

    See Also:
        - PcmData: Standard audio data container (vision_agents.core.types)
        - OutputAudioTrack: Protocol for audio track output (vision_agents.core.edge.types)
        - AUDIO_DOCUMENTATION.md: Comprehensive audio configuration guide
    """

    @abc.abstractmethod
    async def create_user(self, user: User):
        pass

    @abc.abstractmethod
    def create_audio_track(self) -> OutputAudioTrack:
        """Create an audio track for publishing to the transport layer.

        This method creates a new audio track that can be used to publish audio
        data to the transport layer. The specific implementation and default
        audio format (sample rate, channels, bit depth) varies by transport:

        Transport-Specific Defaults:
            - StreamEdge (GetStream): 48000 Hz, 2 channels (stereo), 16-bit
            - LocalEdge (localrtc): 16000 Hz, 1 channel (mono), 16-bit

        Returns:
            OutputAudioTrack: An audio track instance ready for publishing.

        Example:
            # Create audio track with transport defaults
            edge = StreamEdge(connection=conn)
            audio_track = edge.create_audio_track()

            # Publish the track
            await edge.publish_tracks(room, audio_track=audio_track)

        Note:
            Implementations may accept additional parameters for custom
            audio configuration (sample rate, channels, etc.). Check the
            specific transport documentation for available options.

        See Also:
            - StreamEdge.create_audio_track(): GetStream implementation with configurable format
            - AUDIO_DOCUMENTATION.md: Audio format specifications and best practices
        """
        pass

    @abc.abstractmethod
    async def close(self):
        pass

    @abc.abstractmethod
    def open_demo(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    async def join(self, *args, **kwargs) -> Room:
        pass

    @abc.abstractmethod
    async def publish_tracks(self, room: Room, audio_track, video_track):
        pass

    @abc.abstractmethod
    async def create_conversation(self, call: Any, user: User, instructions):
        pass

    @abc.abstractmethod
    def add_track_subscriber(
        self, track_id: str, callback: Callable[[PcmData], None]
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        """Subscribe to a track and receive data via callback.

        This method subscribes to an audio or video track from the transport layer
        and registers a callback to receive the track's data. For audio tracks,
        the callback receives PcmData objects containing standardized PCM audio.

        Args:
            track_id: Identifier for the track to subscribe to (e.g., "audio", "video")
            callback: Function to call when track data is received.
                     For audio tracks, receives PcmData with format information:
                     - sample_rate: Audio sample rate in Hz
                     - channels: Number of audio channels
                     - bit_depth: Bits per sample
                     - data: Raw PCM audio as numpy array
                     - timestamp: Optional Unix timestamp

        Returns:
            The MediaStreamTrack instance for the subscribed track, or None if not found.

        Example:
            def on_audio(pcm_data: PcmData):
                print(f"Received audio: {pcm_data.sample_rate} Hz, {pcm_data.channels} ch")
                print(f"Audio samples: {len(pcm_data.data)}")
                # Process audio data...

            # Subscribe to audio track
            track = edge.add_track_subscriber("audio", on_audio)

        Note:
            Some transport implementations (like GetStream) may deliver audio data
            through alternative mechanisms (e.g., connection events) and may not
            invoke the callback directly. Check implementation-specific documentation.

        See Also:
            - PcmData: Standard audio data type
            - AUDIO_DOCUMENTATION.md: Audio format and processing guide
        """
        pass

    @abc.abstractmethod
    async def create_call(self, call_type: str, call_id: str) -> Room:
        """Create a call/room with the given type and ID.

        This method creates a new call or room instance that can be used
        for joining and managing communication sessions.

        Args:
            call_type: The type of call (e.g., "default", "video", "audio")
            call_id: Unique identifier for the call

        Returns:
            A Room instance representing the created call
        """
        pass
