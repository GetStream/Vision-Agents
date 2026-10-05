import asyncio
import logging
import os
import time
import webbrowser
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode

import aiortc
from vision_agents.core.agents.agents import tracer
from vision_agents.core.edge import Call, EdgeTransport, events
from vision_agents.core.edge.types import Connection, Participant, TrackType, User
from vision_agents.core.utils import get_vision_agents_version
from vision_agents.plugins.getstream.stream_conversation import StreamConversation

import getstream.models
from getstream import AsyncStream
from getstream.exceptions import StreamApiException
from getstream.models import (
    ChannelInput,
    ChannelMemberRequest,
    FullUserResponse,
    UpdateUserPartialRequest,
    UserRequest,
)
from getstream.video import rtc
from getstream.video.async_call import Call as StreamCall
from getstream.video.rtc import AudioStreamTrack, ConnectionManager
from getstream.video.rtc.participants import ParticipantsState
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    TrackType as StreamTrackType,
)
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

from .remote_video_track import RemoteVideoTrack

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)


# Conversion maps and functions for getstream -> core types
_TRACK_TYPE_MAP: dict[int, TrackType] = {
    StreamTrackType.TRACK_TYPE_UNSPECIFIED: TrackType.UNSPECIFIED,
    StreamTrackType.TRACK_TYPE_VIDEO: TrackType.VIDEO,
    StreamTrackType.TRACK_TYPE_AUDIO: TrackType.AUDIO,
    StreamTrackType.TRACK_TYPE_SCREEN_SHARE: TrackType.SCREEN_SHARE,
    StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO: TrackType.SCREEN_SHARE_AUDIO,
}


def _to_core_track_type(stream_track_type: rtc.TrackType) -> TrackType:
    """Convert getstream TrackType to core TrackType."""
    # rtc.TrackType is not hashable; its int is the protobuf TrackType value.
    type_ = _TRACK_TYPE_MAP.get(int(stream_track_type))
    if type_ is None:
        raise ValueError(f"Unknown track type: {stream_track_type}")
    return type_


def _track_id(session_id: str, track_type: rtc.TrackType) -> str:
    """Build the core track id; the SDK keeps one RemoteTrack per session and type."""
    return f"{session_id}:{int(track_type)}"


def _to_core_participant(
    participant: rtc.RemoteParticipant | None,
) -> Participant | None:
    """Convert a getstream RemoteParticipant to core Participant type.

    Args:
        participant: RemoteParticipant from the SDK, or None

    Returns:
        Core Participant with original and user_id, or None
    """
    if participant is None:
        return None

    unique_id = f"{participant.user_id}__{participant.session_id}"
    return Participant(original=participant, user_id=participant.user_id, id=unique_id)


class StreamConnection(Connection):
    def __init__(self, connection: ConnectionManager):
        super().__init__()
        # store the native connection object
        self._connection = connection
        self._idle_since: float = 0.0
        self._participant_joined = asyncio.Event()
        # Subscribe to participants changes for this connection
        self._subscription = self._connection.participants_state.map(
            self._on_participant_change
        )

    @property
    def participants(self) -> ParticipantsState:
        return self._connection.participants_state

    def idle_since(self) -> float:
        """
        Return the timestamp when all participants left this call except the agent itself.
        `0.0` means that connection is active.

        Returns:
            idle time for this connection or 0.
        """
        return self._idle_since

    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        """
        Wait for at least one participant other than the agent to join.
        """
        await asyncio.wait_for(self._participant_joined.wait(), timeout=timeout)

    async def close(self, timeout: float = 2.0):
        try:
            await asyncio.wait_for(self._connection.leave(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Connection leave timed out during close")
        except RuntimeError as e:
            if "asynchronous generator" in str(e):
                logger.debug(f"Ignoring async generator error during shutdown: {e}")
            else:
                raise
        except Exception as e:
            logger.error(f"Error during connection close: {e}")

    def _on_participant_change(self, participants: list[rtc.RemoteParticipant]) -> None:
        # Get all participants except the agent itself.
        other_participants = [
            p for p in participants if p.user_id != self._connection.user_id
        ]
        if other_participants:
            # Some participants detected.
            # Reset the idleness timeout back to zero.
            self._idle_since = 0.0
            # Resolve the participant joined event
            self._participant_joined.set()
        elif not self._idle_since:
            # No participants left, register the time the connection became idle if it's not set.
            self._idle_since = time.time()


class StreamEdge(EdgeTransport[StreamCall]):
    """
    StreamEdge uses getstream.io's edge network. To support multiple vendors, this means we expose

    """

    client: AsyncStream

    def __init__(self, **kwargs):
        # Initialize Stream client
        super().__init__()
        version = get_vision_agents_version()
        self.client = AsyncStream(user_agent=f"stream-vision-agents-{version}")
        # self.events is inherited from EdgeTransport (with required events already registered)
        self.events.register_events_from_module(getstream.models, "call.")
        self.conversation: Optional[StreamConversation] = None
        self.channel_type = "messaging"
        self._agent_user_id: str | None = None
        # Dropping the last reference to a RemoteTrack unsubscribes it, so the
        # tracks are kept until their participant leaves.
        self._remote_tracks: dict[str, rtc.RemoteTrack] = {}
        # Track ids that are unpublished (muted) now; the RemoteTrack resumes after unmute.
        self._unpublished: set[str] = set()
        # Sessions that left the call. The SDK sends tracks and SFU events on
        # separate streams, so a track_added can arrive after participant_left.
        self._left_sessions: set[str] = set()

        self._real_connection: Optional[ConnectionManager] = None
        self._call: Optional[StreamCall] = None

    @property
    def _connection(self) -> ConnectionManager:
        if self._real_connection is None:
            raise ValueError("Edge connection is not set")
        return self._real_connection

    async def _on_track_added(self, track: rtc.RemoteTrack):
        """Keep the remote track and spawn TrackAddedEvent."""
        participant = track.participant
        if participant.session_id in self._left_sessions:
            return
        track_id = _track_id(participant.session_id, track.track_type)
        self._remote_tracks[track_id] = track
        self._unpublished.discard(track_id)
        self.events.send(
            events.TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=_to_core_track_type(track.track_type),
                participant=_to_core_participant(participant),
            )
        )

    async def _on_track_published(self, event: rtc.TrackPublished):
        """Spawn TrackAddedEvent when an unpublished track is published again."""
        track_id = _track_id(event.session_id, event.track_type)
        # The first publish of a track arrives as track_added.
        if track_id not in self._unpublished:
            return
        self._unpublished.discard(track_id)
        self.events.send(
            events.TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=_to_core_track_type(event.track_type),
                participant=_to_core_participant(
                    self._remote_tracks[track_id].participant
                ),
            )
        )

    async def _on_track_unpublished(self, event: rtc.TrackUnpublished):
        """Spawn TrackRemovedEvent and keep the track for a later unmute."""
        track_id = _track_id(event.session_id, event.track_type)
        track = self._remote_tracks.get(track_id)
        if track is None or track_id in self._unpublished:
            return
        self._unpublished.add(track_id)
        logger.info(
            f"Track unpublished: {event.user_id}, "
            f"track: {StreamTrackType.Name(int(event.track_type))}"
        )
        self.events.send(
            events.TrackRemovedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=_to_core_track_type(event.track_type),
                participant=_to_core_participant(track.participant),
            )
        )

    async def _on_call_ended(self, event: rtc.CallEnded):
        if self._call is None:
            return
        self.events.send(
            events.CallEndedEvent(
                plugin_name="getstream",
                call=self._call,
            )
        )

    async def _on_sfu_participant_joined(self, event: rtc.ParticipantJoined):
        core_participant = _to_core_participant(event.participant)
        if core_participant is None:
            return
        if core_participant.user_id == self._agent_user_id:
            return
        if self._call is None:
            return
        self.events.send(
            events.ParticipantJoinedEvent(
                plugin_name="getstream",
                participant=core_participant,
                call=self._call,
            )
        )

    async def _on_sfu_participant_left(self, event: rtc.ParticipantLeft):
        participant = event.participant
        self._left_sessions.add(participant.session_id)
        track_ids = [
            track_id
            for track_id, track in self._remote_tracks.items()
            if track.participant.session_id == participant.session_id
        ]
        logger.info(f"Participant left: {participant.user_id}, tracks: {track_ids}")
        for track_id in track_ids:
            track = self._remote_tracks.pop(track_id)
            if track_id in self._unpublished:
                # TrackRemovedEvent was already sent at unpublish.
                self._unpublished.discard(track_id)
                continue
            self.events.send(
                events.TrackRemovedEvent(
                    plugin_name="getstream",
                    track_id=track_id,
                    track_type=_to_core_track_type(track.track_type),
                    participant=_to_core_participant(participant),
                )
            )

        core_participant = _to_core_participant(participant)
        if core_participant is None:
            return
        if core_participant.user_id == self._agent_user_id:
            return
        if self._call is None:
            return
        self.events.send(
            events.ParticipantLeftEvent(
                plugin_name="getstream",
                participant=core_participant,
                call=self._call,
            )
        )

    async def create_conversation(self, call: Call, user: User, instructions: str):
        channel = self.client.chat.channel(self.channel_type, call.id)
        await channel.get_or_create(
            data=ChannelInput(created_by_id=user.id),
        )
        self.conversation = StreamConversation(instructions, [], channel)
        return self.conversation

    def _require_authenticated(self) -> None:
        if self._agent_user_id is None:
            raise RuntimeError(
                "StreamEdge is not authenticated. Call authenticate() first."
            )

    async def authenticate(self, user: User) -> None:
        await self._merge_user(user)
        self._agent_user_id = user.id

    async def create_users(self, users: list[User]) -> list[FullUserResponse]:
        """Create multiple users, one API call per user."""

        return await asyncio.gather(*(self._merge_user(u) for u in users))

    async def _merge_user(self, user: User) -> FullUserResponse:
        """Create the user, or merge the locally set fields into an existing one.

        Upserting replaces the stored user, so custom data written by other
        clients would be lost. A partial update merges instead; it only fails
        when the user does not exist yet, in which case we create it.

        A custom field set to None is removed from the stored user rather than
        written as null.
        """
        custom: dict[str, object] = {
            key: value for key, value in user.custom.items() if value is not None
        }
        unset: list[str] = [key for key, value in user.custom.items() if value is None]
        set_fields: dict[str, object] = {
            key: value
            for key, value in (("name", user.name), ("image", user.image))
            if value
        }
        set_fields.update(custom)
        try:
            response = await self.client.update_users_partial(
                [UpdateUserPartialRequest(id=user.id, set=set_fields, unset=unset)]
            )
        except StreamApiException as e:
            if e.status_code != 404:
                raise
            response = await self.client.upsert_users(
                UserRequest(id=user.id, name=user.name, image=user.image, custom=custom)
            )
        return response.data.users[user.id]

    async def create_call(self, call_id: str, **kwargs) -> StreamCall:
        """Shortcut for creating a call/room etc."""
        self._require_authenticated()
        call_type = kwargs.get("call_type", "default")
        call = self.client.video.call(call_type, call_id)
        await call.get_or_create(data={"created_by_id": self._agent_user_id})
        return call

    async def join(
        self, agent: "Agent", call: StreamCall, **kwargs
    ) -> StreamConnection:
        """Join a GetStream call and establish a WebRTC connection.

        This method:
        - Configures WebRTC subscription for audio/video tracks
        - Joins the call with the agent's user ID
        - Maps participant, track, audio and call events to core events
        - Establishes the connection

        Args:
            agent: The Agent instance joining the call.
            call: StreamCall object representing the GetStream call to join.
            **kwargs: Additional configuration options (unused).

        Returns:
            StreamConnection: A connection wrapper implementing the core Connection interface.
        """

        # Traditional mode - use WebRTC connection
        # Configure subscription for audio and video
        subscription_config = SubscriptionConfig(
            default=self._get_subscription_config()
        )

        # Open RTC connection and keep it alive for the duration of the returned context manager
        connection = await rtc.join(
            call,
            agent.agent_user.id,
            subscription_config=subscription_config,
        )
        # Store immediately so close() can clean up if join is interrupted
        self._real_connection = connection

        self.events.silent(events.AudioReceivedEvent)

        @connection.on("audio")
        async def on_audio_received(pcm: PcmData):
            self.events.send(
                events.AudioReceivedEvent(
                    plugin_name="getstream",
                    pcm_data=pcm,
                    participant=_to_core_participant(pcm.participant),
                )
            )

        # Registered before the connection starts: the SDK sends participant_joined
        # and track_added for the participants who are already in the call.
        connection.on("track_added", self._on_track_added)
        connection.on("track_published", self._on_track_published)
        connection.on("track_unpublished", self._on_track_unpublished)
        connection.on("participant_joined", self._on_sfu_participant_joined)
        connection.on("participant_left", self._on_sfu_participant_left)
        connection.on("call_ended", self._on_call_ended)

        self._call = call
        # Start the connection
        await connection.__aenter__()

        standardize_connection = StreamConnection(connection)
        return standardize_connection

    def create_audio_track(
        self, sample_rate: int = 48000, stereo: bool = False
    ) -> AudioStreamTrack:
        return AudioStreamTrack(
            audio_buffer_size_ms=300_000,
            sample_rate=sample_rate,
            channels=stereo and 2 or 1,
        )  # default to webrtc framerate

    def add_track_subscriber(self, track_id: str) -> Optional[aiortc.VideoStreamTrack]:
        track = self._remote_tracks.get(track_id)
        if track is None:
            return None
        # Each call gets an independent frame stream.
        return RemoteVideoTrack(track.video_frames())

    async def publish_tracks(
        self,
        audio_track: Optional[aiortc.MediaStreamTrack],
        video_track: Optional[aiortc.MediaStreamTrack],
    ):
        """
        Add the tracks to publish audio and video
        """
        await self._connection.add_tracks(audio=audio_track, video=video_track)
        if audio_track:
            logger.info("🤖 Agent ready to speak")
        if video_track:
            logger.info("🎥 Agent ready to publish video")
        # In Realtime mode we directly publish the provider's output track; no extra forwarding needed

    def _get_subscription_config(self):
        return TrackSubscriptionConfig(
            track_types=[
                StreamTrackType.TRACK_TYPE_VIDEO,
                StreamTrackType.TRACK_TYPE_AUDIO,
                StreamTrackType.TRACK_TYPE_SCREEN_SHARE,
                StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
            ]
        )

    async def close(self):
        if self._real_connection:
            try:
                await self._real_connection.leave()
            except Exception:
                logger.exception("Error during connection leave")
            self._real_connection = None
        self._remote_tracks.clear()
        self._unpublished.clear()
        self._left_sessions.clear()
        if self.client:
            await self.client.aclose()
        self._call = None

    async def send_custom_event(self, data: dict) -> None:
        """Send a custom event to all participants watching the call.

        Custom events are delivered to clients subscribed to the call via
        `call.on("custom", callback)`. The payload is limited to 5KB.

        Args:
            data: Custom event payload (must be JSON-serializable).

        Raises:
            RuntimeError: If not connected to a call.
        """
        if self._call is None:
            raise RuntimeError("Cannot send custom event: not connected to a call")
        self._require_authenticated()
        await self._call.send_call_event(user_id=self._agent_user_id, custom=data)

    @tracer.start_as_current_span("stream_edge.open_demo")
    async def open_demo_for_agent(
        self, agent: "Agent", call_type: str, call_id: str
    ) -> str:
        call = await agent.create_call(call_type, call_id)

        return await self.open_demo(call)

    @tracer.start_as_current_span("stream_edge.open_demo")
    async def open_demo(self, call: StreamCall) -> str:
        client = call.client.stream

        # Create a human user for testing
        human_id = "user-demo-agent"
        name = "Human User"

        # Create the user in the GetStream system
        await client.create_user(name=name, id=human_id)

        # Ensure that both agent and user get access the demo by adding the user as member and the agent the channel creator
        channel = client.chat.channel(self.channel_type, call.id)
        # Ensure the agent user is authenticated before creating the channel
        self._require_authenticated()
        response = await channel.get_or_create(
            data=ChannelInput(
                created_by_id=self._agent_user_id,
                members=[
                    ChannelMemberRequest(
                        user_id=human_id,
                    )
                ],
            )
        )

        if human_id not in [m.user_id for m in response.data.members]:
            await channel.update(add_members=[ChannelMemberRequest(user_id=human_id)])

        # Create user token for browser access
        token = client.create_token(human_id, expiration=3600)

        """Helper function to open browser with Stream call link."""
        base_url = (
            f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
        )
        params = {
            "api_key": client.api_key,
            "token": token,
            "skip_lobby": "true",
            "user_name": name,
            "video_encoder": "h264",  # Use H.264 instead of VP8 for better compatibility
            "bitrate": 12000000,
            "w": 1920,
            "h": 1080,
            "channel_type": self.channel_type,
        }

        url = f"{base_url}{call.id}?{urlencode(params)}"
        logger.info(f"🌐 Opening browser to: {url}")

        try:
            # Run webbrowser.open in a separate thread to avoid blocking the event loop
            await asyncio.to_thread(webbrowser.open, url)
            logger.info("✅ Browser opened successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to open browser: {e}")
            logger.warning(f"Please manually open this URL: {url}")

        return url
