import asyncio
import datetime
import logging
import os
import time
import webbrowser
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlencode

import aiortc
from getstream import AsyncStream
from getstream.chat.async_client import ChatClient
from getstream.models import ChannelInput, ChannelMember, ChannelMemberRequest
from getstream.video import rtc
from getstream.video.async_call import Call
from getstream.video.rtc import ConnectionManager, audio_track
from getstream.video.rtc.participants import ParticipantsState
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import (
    Participant,
    TrackType,
)
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from vision_agents.core.agents.agents import tracer
from vision_agents.core.edge import EdgeTransport, events, sfu_events
from vision_agents.core.edge.types import Connection, OutputAudioTrack, User
from vision_agents.core.events.manager import EventManager
from vision_agents.core.utils import get_vision_agents_version
from vision_agents.plugins.getstream.stream_conversation import StreamConversation

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)


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

    def _on_participant_change(self, participants: list[Participant]) -> None:
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


class StreamEdge(EdgeTransport):
    """
    StreamEdge uses getstream.io's edge network. To support multiple vendors, this means we expose

    """

    client: AsyncStream

    def __init__(self, **kwargs):
        # Initialize Stream client
        super().__init__()
        version = get_vision_agents_version()
        self.client = AsyncStream(user_agent=f"vision-agents-{version}")
        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.events.register_events_from_module(sfu_events)
        self.conversation: Optional[StreamConversation] = None
        self.channel_type = "messaging"
        self.agent_user_id: str | None = None
        # Track mapping: (user_id, session_id, track_type_int) -> {"track_id": str, "published": bool}
        # track_type_int is from TrackType enum (e.g., TrackType.TRACK_TYPE_AUDIO)
        self._track_map: dict = {}
        # Temporary storage for tracks before SFU confirms their type
        # track_id -> (user_id, session_id, webrtc_type_string)
        self._pending_tracks: dict = {}

        self._real_connection: Optional[ConnectionManager] = None

        # Register event handlers
        self.events.subscribe(self._on_track_published)
        self.events.subscribe(self._on_track_removed)
        self.events.subscribe(self._on_call_ended)

    @property
    def _connection(self) -> ConnectionManager:
        if self._real_connection is None:
            raise ValueError("Edge connection is not set")
        return self._real_connection

    def _get_webrtc_kind(self, track_type_int: int) -> str:
        """Get the expected WebRTC kind (audio/video) for a SFU track type."""
        # Map SFU track types to WebRTC kinds
        if track_type_int in (
            TrackType.TRACK_TYPE_AUDIO,
            TrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
        ):
            return "audio"
        elif track_type_int in (
            TrackType.TRACK_TYPE_VIDEO,
            TrackType.TRACK_TYPE_SCREEN_SHARE,
        ):
            return "video"
        else:
            # Default to video for unknown types
            return "video"

    async def _on_track_published(self, event: sfu_events.TrackPublishedEvent):
        """Handle track published events from SFU - spawn TrackAddedEvent with correct type."""
        if not event.payload:
            return

        if event.participant and event.participant.user_id:
            session_id = event.participant.session_id
            user_id = event.participant.user_id
        else:
            user_id = event.payload.user_id
            session_id = event.payload.session_id

        track_type_int = event.payload.type  # TrackType enum int from SFU
        expected_kind = self._get_webrtc_kind(track_type_int)
        track_key = (user_id, session_id, track_type_int)
        is_agent_track = user_id == self.agent_user_id

        # Skip processing the agent's own tracks - we don't subscribe to them
        if is_agent_track:
            logger.debug(f"Skipping agent's own track: {track_type_int} from {user_id}")
            return

        # First check if track already exists in map (e.g., from previous unpublish/republish)
        if track_key in self._track_map:
            self._track_map[track_key]["published"] = True
            track_id = self._track_map[track_key]["track_id"]

            # Emit TrackAddedEvent so agent can switch to this track
            self.events.send(
                events.TrackAddedEvent(
                    plugin_name="getstream",
                    track_id=track_id,
                    track_type=track_type_int,
                    user=event.participant,
                )
            )
            return

        # Wait for pending track to be populated (with 10 second timeout)
        # SFU might send TrackPublishedEvent before WebRTC processes track_added
        track_id = None
        timeout = 10.0
        poll_interval = 0.01
        elapsed = 0.0

        while elapsed < timeout:
            # Find pending track for this user/session with matching kind
            for tid, (pending_user, pending_session, pending_kind) in list(
                self._pending_tracks.items()
            ):
                if (
                    pending_user == user_id
                    and pending_session == session_id
                    and pending_kind == expected_kind
                ):
                    track_id = tid
                    del self._pending_tracks[tid]
                    break

            if track_id:
                break

            # Wait a bit before checking again
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        if track_id:
            # Store with correct type from SFU
            self._track_map[track_key] = {"track_id": track_id, "published": True}

            # Only emit TrackAddedEvent for remote participants, not for agent's own tracks
            if not is_agent_track:
                # NOW spawn TrackAddedEvent with correct type
                self.events.send(
                    events.TrackAddedEvent(
                        plugin_name="getstream",
                        track_id=track_id,
                        track_type=track_type_int,
                        user=event.participant,
                        participant=event.participant,
                    )
                )

        else:
            raise TimeoutError(
                f"Timeout waiting for pending track: {track_type_int} ({expected_kind}) from user {user_id}, "
                f"session {session_id}. Waited {timeout}s but WebRTC track_added with matching kind was never received."
                f"Pending tracks: {self._pending_tracks}\n"
                f"Key: {track_key}\n"
                f"Track map: {self._track_map}\n"
            )

    async def _on_track_removed(
        self, event: sfu_events.ParticipantLeftEvent | sfu_events.TrackUnpublishedEvent
    ):
        """Handle track unpublished and participant left events."""
        if not event.payload:  # NOTE: mypy typecheck
            return

        participant = event.participant
        if participant and participant.user_id:
            user_id = participant.user_id
            session_id = participant.session_id
        else:
            user_id = event.payload.user_id
            session_id = event.payload.session_id

        # Determine which tracks to remove
        if hasattr(event.payload, "type") and event.payload is not None:
            # TrackUnpublishedEvent - single track
            tracks_to_remove = [event.payload.type]
            event_desc = "Track unpublished"
        else:
            # ParticipantLeftEvent - all published tracks
            tracks_to_remove = (
                event.participant.published_tracks if event.participant else None
            ) or []
            event_desc = "Participant left"

        track_names = [TrackType.Name(t) for t in tracks_to_remove]
        logger.info(f"{event_desc}: {user_id}, tracks: {track_names}")

        # Mark each track as unpublished and send TrackRemovedEvent
        for track_type_int in tracks_to_remove:
            track_key = (user_id, session_id, track_type_int)
            track_info = self._track_map.get(track_key)

            if track_info:
                track_id = track_info["track_id"]
                self.events.send(
                    events.TrackRemovedEvent(
                        plugin_name="getstream",
                        track_id=track_id,
                        track_type=track_type_int,
                        user=participant,
                        # TODO: user=participant?
                        participant=participant,
                    )
                )
                # Mark as unpublished instead of removing
                self._track_map[track_key]["published"] = False
            else:
                logger.warning(f"Track not found in map: {track_key}")

    async def _on_call_ended(self, event: sfu_events.CallEndedEvent):
        self.events.send(
            events.CallEndedEvent(
                plugin_name="getstream",
            )
        )

    async def create_conversation(self, call: Call, user, instructions):
        chat_client: ChatClient = call.client.stream.chat
        channel = chat_client.channel(self.channel_type, call.id)
        await channel.get_or_create(
            data=ChannelInput(created_by_id=user.id),
        )
        self.conversation = StreamConversation(instructions, [], channel)
        return self.conversation

    async def create_user(self, user: User):
        self.agent_user_id = user.id
        return await self.client.create_user(
            name=user.name, id=user.id, image=user.image
        )

    async def create_users(self, users: list[User]):
        """Create multiple users in a single API call."""
        from getstream.models import UserRequest

        users_map = {u.id: UserRequest(name=u.name, id=u.id) for u in users}
        response = await self.client.update_users(users_map)
        return [response.data.users[u.id] for u in users]

    async def join(self, agent: "Agent", call: Call) -> StreamConnection:
        """
        The logic for joining a call is different for each edge network/realtime audio/video provider

        This function
        - initializes the chat channel
        - has the agent.agent_user join the call
        - connects incoming audio/video to the agent
        - connecting agent's outgoing audio/video to the call
        """

        # Traditional mode - use WebRTC connection
        # Configure subscription for audio and video
        subscription_config = SubscriptionConfig(
            default=self._get_subscription_config()
        )

        # Open RTC connection and keep it alive for the duration of the returned context manager
        connection = await rtc.join(
            call, agent.agent_user.id, subscription_config=subscription_config
        )

        @connection.on("track_added")
        async def on_track(track_id, track_type, user):
            # Store track in pending map - wait for SFU to confirm type before spawning TrackAddedEvent
            self._pending_tracks[track_id] = (user.user_id, user.session_id, track_type)

        self.events.silent(events.AudioReceivedEvent)

        @connection.on("audio")
        async def on_audio_received(pcm: PcmData):
            self.events.send(
                events.AudioReceivedEvent(
                    plugin_name="getstream",
                    pcm_data=pcm,
                    participant=pcm.participant,
                )
            )

        # Re-emit certain events from the underlying RTC stack
        # for the Agent to subscribe.
        connection.on("participant_joined", self.events.send)
        connection.on("participant_left", self.events.send)
        connection.on("track_published", self.events.send)
        connection.on("track_unpublished", self.events.send)
        connection.on("call_ended", self.events.send)

        # Start the connection
        await connection.__aenter__()
        # Re-publish already published tracks in case somebody is already on the call when we joined.
        # Otherwise, we won't get the video track from participants joined before us.
        await connection.republish_tracks()
        self._real_connection = connection

        standardize_connection = StreamConnection(connection)
        return standardize_connection

    def create_audio_track(
        self, framerate: int = 48000, stereo: bool = True
    ) -> OutputAudioTrack:
        return audio_track.AudioStreamTrack(
            audio_buffer_size_ms=300_000,
            sample_rate=framerate,
            channels=stereo and 2 or 1,
        )  # default to webrtc framerate

    def create_video_track(self):
        return aiortc.VideoStreamTrack()

    def add_track_subscriber(
        self, track_id: str
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        return self._connection.subscriber_pc.add_track_subscriber(track_id)

    async def publish_tracks(self, audio_track, video_track):
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
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_AUDIO,
                TrackType.TRACK_TYPE_SCREEN_SHARE,
                TrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
            ]
        )

    async def close(self):
        # Note: Not calling super().close() as it's an abstract method with trivial body
        pass

    @tracer.start_as_current_span("stream_edge.open_demo")
    async def open_demo_for_agent(
        self, agent: "Agent", call_type: str, call_id: str
    ) -> str:
        await agent.create_user()
        call = await agent.create_call(call_type, call_id)

        return await self.open_demo(call)

    @tracer.start_as_current_span("stream_edge.open_demo")
    async def open_demo(self, call: Call) -> str:
        client = call.client.stream

        # Create a human user for testing
        human_id = "user-demo-agent"
        name = "Human User"

        # Create the user in the GetStream system
        await client.create_user(name=name, id=human_id)

        # Ensure that both agent and user get access the demo by adding the user as member and the agent the channel creator
        channel = client.chat.channel(self.channel_type, call.id)
        response = await channel.get_or_create(
            data=ChannelInput(
                created_by_id=self.agent_user_id,
                members=[
                    ChannelMemberRequest(
                        user_id=human_id,
                    )
                ],
            )
        )

        if human_id not in [m.user_id for m in response.data.members]:
            await channel.update(
                add_members=[
                    ChannelMember(
                        user_id=human_id,
                        # TODO: get rid of this when codegen for stream-py is fixed, these fields are meaningless
                        banned=False,
                        channel_role="",
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        notifications_muted=False,
                        shadow_banned=False,
                        updated_at=datetime.datetime.now(datetime.timezone.utc),
                        custom={},
                    )
                ]
            )

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
