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
    async def create_user(self, user: User) -> None:
        """Create a user in the edge transport.

        **Required method** - Must be implemented by all transport implementations.

        This method initializes a user in the transport layer. The specific
        implementation varies by transport:
        - StreamEdge: Creates user in GetStream service
        - LocalEdge: Stores user locally for session management

        Args:
            user: User object containing user information (id, name, image).

        Example:
            >>> edge = LocalEdge()
            >>> user = User(id="user-123", name="Alice")
            >>> await edge.create_user(user)
        """
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
    async def close(self) -> None:
        """Close the edge transport and clean up resources.

        **Required method** - Must be implemented by all transport implementations.

        This method performs cleanup operations including:
        - Stopping audio/video capture
        - Closing network connections
        - Releasing device resources
        - Clearing internal state

        Should be called when the transport is no longer needed to prevent
        resource leaks.

        Example:
            >>> edge = LocalEdge()
            >>> # ... use edge ...
            >>> await edge.close()  # Clean up when done
        """
        pass

    @abc.abstractmethod
    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """Open a demo session.

        **Optional method** - Implementation can be a no-op for transports
        that don't support demo mode (e.g., LocalEdge).

        This method is used by transports that provide web-based demo interfaces
        (e.g., StreamEdge opens GetStream's demo UI). Implementations that don't
        provide a demo interface should implement this as a no-op.

        Args:
            *args: Positional arguments for demo configuration.
            **kwargs: Keyword arguments for demo configuration.

        Note:
            LocalEdge implements this as a no-op since local transports don't
            have a web-based demo interface.

        Example:
            >>> edge = StreamEdge(connection=conn)
            >>> edge.open_demo()  # Opens browser to GetStream demo UI
            >>>
            >>> edge = LocalEdge()
            >>> edge.open_demo()  # No-op for local transport
        """
        pass

    @abc.abstractmethod
    async def join(self, *args: Any, **kwargs: Any) -> Room:
        """Join a room and start device capture/streaming.

        **Required method** - Must be implemented by all transport implementations.

        This method connects to a room and begins capturing/streaming audio and
        video. The specific implementation varies by transport:

        - **LocalEdge**: Creates local room and starts device capture
        - **StreamEdge**: Joins GetStream room and publishes WebRTC tracks

        **Calling Convention:**
            The recommended convention is to pass an Agent instance as the first
            positional argument to enable audio format negotiation:

            >>> room = await edge.join(agent, room_id="call-1", room_type="default")

            Legacy convention without agent is still supported but may not
            negotiate optimal audio formats:

            >>> room = await edge.join(room_id="call-1", room_type="default")

        Args:
            *args: Positional arguments. First argument should be Agent instance
                for audio format negotiation (recommended).
            **kwargs: Keyword arguments for join configuration.
                Common kwargs:
                - room_id (str): Unique identifier for the room
                - room_type (str): Type of room (e.g., "default", "audio", "video")

        Returns:
            A Room instance representing the joined room.

        Example:
            >>> edge = LocalEdge(audio_device="default")
            >>> agent = Agent(edge=edge, llm=gemini.Realtime())
            >>> room = await edge.join(agent, room_id="my-call", room_type="default")
            >>> # Room is now active, devices are capturing
        """
        pass

    @abc.abstractmethod
    async def publish_tracks(
        self, room: Room, audio_track: Any = None, video_track: Any = None
    ) -> None:
        """Publish audio and video tracks to the room.

        **Required method** - Must be implemented by all transport implementations.

        This method publishes media tracks to the room, making them available
        to other participants. Tracks can be published individually or together.

        **Calling Convention:**
            The recommended convention is to use keyword arguments with explicit
            room parameter:

            >>> await edge.publish_tracks(room, audio_track=audio, video_track=video)
            >>> await edge.publish_tracks(room, audio_track=audio)  # Audio only

        Args:
            room: The room to publish tracks to. Must be a Room instance.
            audio_track: Audio track to publish (e.g., AudioInputTrack, AudioOutputTrack).
                Optional - only required if publishing audio.
            video_track: Video track to publish (e.g., VideoInputTrack).
                Optional - only required if publishing video.

        Note:
            Some implementations may support legacy calling conventions without
            explicit room parameter. These are deprecated and will be removed in
            future versions.

        Example:
            >>> edge = LocalEdge()
            >>> room = await edge.join(agent, room_id="my-call")
            >>> audio = AudioInputTrack(device="default", sample_rate=16000)
            >>> video = VideoInputTrack(device=0)
            >>> await edge.publish_tracks(room, audio_track=audio, video_track=video)
        """
        pass

    @abc.abstractmethod
    async def create_conversation(self, call: Any, user: User, instructions: Any) -> None:
        """Create a conversation in the call.

        **Optional method** - Implementation can be a no-op for transports
        that don't manage conversations externally (e.g., LocalEdge).

        This method is used by transports that manage conversation state
        externally (e.g., StreamEdge creates conversations in GetStream).
        For transports that manage conversations locally or through the LLM
        directly, this can be a no-op.

        Args:
            call: The call object to create the conversation in.
            user: User object representing the participant.
            instructions: Conversation instructions or configuration.

        Note:
            LocalEdge implements this as a no-op since conversations are
            managed by the LLM provider (e.g., Gemini), not the transport layer.

        Example:
            >>> edge = StreamEdge(connection=conn)
            >>> call = await edge.create_call("default", "my-call")
            >>> user = User(id="user-123", name="Alice")
            >>> await edge.create_conversation(call, user, instructions="Be helpful")
            >>>
            >>> edge = LocalEdge()
            >>> await edge.create_conversation(call, user, instructions)  # No-op
        """
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

        **Required method** - Must be implemented by all transport implementations.

        This method creates a new call or room instance that can be used
        for joining and managing communication sessions. The specific
        implementation varies by transport:

        - **LocalEdge**: Creates a lightweight LocalRoom instance for local sessions
        - **StreamEdge**: Creates a GetStream call/room in the cloud

        Args:
            call_type: The type of call. Common values:
                - "default": Standard call
                - "audio": Audio-only call
                - "video": Video call
                The interpretation of call_type is transport-specific.
            call_id: Unique identifier for the call. This should be globally
                unique within your application to avoid conflicts.

        Returns:
            A Room instance representing the created call. The Room provides
            properties like `id`, `type`, and methods like `leave()`.

        Example:
            >>> edge = LocalEdge()
            >>> room = await edge.create_call("default", "my-call-123")
            >>> print(f"Created room: {room.id} of type {room.type}")
        """
        pass
