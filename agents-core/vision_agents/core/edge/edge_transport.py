"""
EdgeTransport: Abstract base class for media transport implementations.
"""

import abc
from typing import TYPE_CHECKING, Any, Optional

import aiortc
from pyee.asyncio import AsyncIOEventEmitter

from vision_agents.core.edge.types import User, OutputAudioTrack

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent


class EdgeTransport(AsyncIOEventEmitter, abc.ABC):
    """
    Abstract base class for media transport implementations.

    EdgeTransport handles the connection between an AI agent and remote participants,
    managing both incoming media (from participants) and outgoing media (from the agent).

    Audio Flow:
        INPUT (Participants -> Agent):
            - connect() starts receiving audio, delivered via AudioReceivedEvent
            - subscribe_to_track() for video tracks (audio is automatic)

        OUTPUT (Agent -> Participants):
            - create_output_audio_track() creates the track
            - publish_tracks() starts sending to participants

    Lifecycle:
        1. register_user() - Register the agent's identity with the provider
        2. connect() - Establish connection, START RECEIVING audio/video
        3. publish_tracks() - START SENDING agent's audio/video
        4. ... agent runs ...
        5. disconnect() - Clean up and leave
    """

    @abc.abstractmethod
    async def connect(self, agent: "Agent", call: Any):
        """
        Connect to a call or room.

        Audio Direction: INPUT - After this call, audio from participants will
        be delivered via AudioReceivedEvent on the transport's event manager.

        This establishes the WebRTC/media connection and begins receiving
        audio and video from remote participants.

        Args:
            agent: The Agent instance joining the call
            call: Provider-specific call/room object (or None for local transport)

        Returns:
            Connection object for managing the session lifecycle
        """

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the call and release all resources.

        Audio Direction: Stops both INPUT and OUTPUT streams.

        Stops all media streams and closes the connection.
        Safe to call multiple times.
        """

    @abc.abstractmethod
    async def publish_tracks(
        self,
        audio_track: Optional[OutputAudioTrack],
        video_track: Optional[Any],
    ) -> None:
        """
        Publish the agent's media tracks to participants.

        Audio Direction: OUTPUT - After this call, audio written to the
        audio_track will be sent to participants (played on their speakers).

        Args:
            audio_track: Track created by create_output_audio_track(), or None
            video_track: Video track to publish, or None
        """

    @abc.abstractmethod
    def create_output_audio_track(
        self,
        framerate: int = 48000,
        stereo: bool = True,
    ) -> OutputAudioTrack:
        """
        Create an audio track for the agent's outgoing audio.

        Audio Direction: OUTPUT - The returned track is where the agent writes
        TTS/speech audio. Call publish_tracks() to start sending to participants.

        Args:
            framerate: Sample rate in Hz (default: 48000)
            stereo: True for stereo, False for mono

        Returns:
            OutputAudioTrack that accepts write() calls with PCM data
        """

    @abc.abstractmethod
    def subscribe_to_track(
        self, track_id: str
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        """
        Subscribe to receive a remote participant's video track.

        Audio Direction: INPUT (video only) - Audio from participants is received
        automatically via AudioReceivedEvent after connect(). This method is
        specifically for subscribing to VIDEO tracks.

        Args:
            track_id: The track ID from TrackAddedEvent

        Returns:
            MediaStreamTrack for consuming the video, or None if unavailable
        """

    @abc.abstractmethod
    async def register_user(self, user: User) -> None:
        """
        Register the agent's user identity with the provider.

        Audio Direction: N/A - Setup only, no audio handling.

        Some providers require user registration before joining calls.
        For local transport, this is typically a no-op.

        Args:
            user: The agent's user information (id, name, image)
        """

    @abc.abstractmethod
    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """
        Open a demo UI for testing (provider-specific).

        Audio Direction: N/A - UI helper only.

        For cloud providers, this typically opens a browser to a test page.
        For local transport, this may be a no-op.
        """

    @abc.abstractmethod
    async def create_chat_channel(
        self, call: Any, user: User, instructions: str
    ) -> Optional[Any]:
        """
        Create a text chat channel associated with the call.

        Audio Direction: N/A - Text chat only, no audio handling.

        Used for text-based conversation history alongside voice.
        Returns None if text chat is not supported.

        Args:
            call: The call/room object
            user: The agent's user
            instructions: System instructions for the conversation

        Returns:
            Conversation object for text chat, or None
        """
