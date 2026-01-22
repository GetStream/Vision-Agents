import asyncio
import datetime
import logging
import os
import time
import webbrowser
from typing import TYPE_CHECKING, Callable, Optional
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
    TrackType as StreamTrackType,
)
from getstream.video.rtc.track_util import PcmData as StreamPcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
from vision_agents.core.agents.agents import tracer
from vision_agents.core.edge import EdgeTransport, events, sfu_events
from vision_agents.core.edge.types import Connection, OutputAudioTrack, User
from vision_agents.core.events.manager import EventManager
from vision_agents.core.types import PcmData, TrackType
from vision_agents.core.utils import get_vision_agents_version
from vision_agents.plugins.getstream.adapters import adapt_pcm_data, adapt_track_type
from vision_agents.plugins.getstream.stream_conversation import StreamConversation

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)


class StreamConnection(Connection):
    def __init__(self, connection: ConnectionManager, call_id: str, call_type: str):
        super().__init__()
        # store the native connection object
        self._connection = connection
        self._call_id = call_id
        self._call_type = call_type
        self._idle_since: float = 0.0
        self._participant_joined = asyncio.Event()
        # Subscribe to participants changes for this connection
        self._subscription = self._connection.participants_state.map(
            self._on_participant_change
        )

    @property
    def id(self) -> str:
        """Unique identifier for the room/call (Room protocol)."""
        return self._call_id

    @property
    def type(self) -> str:
        """Type or category of the room/call (Room protocol)."""
        return self._call_type

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

    async def leave(self) -> None:
        """Asynchronously leave/disconnect from the call (Room protocol)."""
        await self.close()

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
        # track_type_int is from StreamTrackType enum (e.g., StreamTrackType.TRACK_TYPE_AUDIO)
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
            StreamTrackType.TRACK_TYPE_AUDIO,
            StreamTrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO,
        ):
            return "audio"
        elif track_type_int in (
            StreamTrackType.TRACK_TYPE_VIDEO,
            StreamTrackType.TRACK_TYPE_SCREEN_SHARE,
        ):
            return "video"
        else:
            # Default to video for unknown types
            return "video"

    def _convert_track_type(self, stream_track_type_int: int) -> TrackType:
        """Convert GetStream TrackType integer to core TrackType enum.

        Deprecated: Use adapt_track_type() from adapters module instead.
        This method delegates to the adapter for backwards compatibility.
        """
        return adapt_track_type(stream_track_type_int)

    def _convert_pcm_data(self, stream_pcm: StreamPcmData) -> PcmData:
        """Convert GetStream PcmData to core PcmData.

        Deprecated: Use adapt_pcm_data() from adapters module instead.
        This method delegates to the adapter for backwards compatibility.
        """
        return adapt_pcm_data(stream_pcm)

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
                    track_type=self._convert_track_type(track_type_int),
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
                        track_type=self._convert_track_type(track_type_int),
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

        track_names = [StreamTrackType.Name(t) for t in tracks_to_remove]
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
                        track_type=self._convert_track_type(track_type_int),
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

    async def join(self, agent: "Agent", call: Call):
        """
        The logic for joining a call is different for each edge network/realtime audio/video provider

        This function
        - initializes the chat channel
        - has the agent.agent_user join the call
        - connects incoming audio/video to the agent
        - connecting agent's outgoing audio/video to the call

        Returns:
            StreamConnection: A connection object that satisfies the Room protocol.
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
        async def on_audio_received(pcm: StreamPcmData):
            self.events.send(
                events.AudioReceivedEvent(
                    plugin_name="getstream",
                    pcm_data=self._convert_pcm_data(pcm),
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

        standardize_connection = StreamConnection(connection, call.id, call.call_type)
        return standardize_connection

    def create_audio_track(
        self, framerate: int = 48000, stereo: bool = True
    ) -> OutputAudioTrack:
        """Create an audio track for publishing to the GetStream connection.

        This method creates an AudioStreamTrack configured for the GetStream transport layer.
        The default configuration (48kHz stereo) is optimized for high-quality audio streaming
        over WebRTC.

        Audio Format Specifications:
            - Default sample rate: 48000 Hz (CD-quality, WebRTC standard)
            - Default channels: 2 (stereo) or 1 (mono if stereo=False)
            - Bit depth: 16-bit signed integer (hardcoded in AudioStreamTrack)
            - Buffer size: 300 seconds (300,000 ms) for long-running streams

        Args:
            framerate: Audio sample rate in Hz. Default is 48000 (WebRTC standard).
                      Common values: 8000, 16000, 24000, 48000
            stereo: If True, creates stereo (2-channel) track. If False, creates mono (1-channel).
                   Default is True.

        Returns:
            OutputAudioTrack: An AudioStreamTrack instance ready for publishing.

        Example:
            # Create default high-quality audio track (48kHz stereo)
            audio_track = edge.create_audio_track()

            # Create voice-optimized track (16kHz mono)
            voice_track = edge.create_audio_track(framerate=16000, stereo=False)

        Note:
            The AudioStreamTrack will automatically handle format conversion if needed.
            For voice applications (ASR/TTS), consider using 16kHz mono for better
            compatibility and reduced bandwidth.

        See Also:
            - AUDIO_DOCUMENTATION.md for comprehensive audio configuration guide
            - AudioStreamTrack class for internal implementation details
        """
        return audio_track.AudioStreamTrack(
            audio_buffer_size_ms=300_000,
            sample_rate=framerate,
            channels=stereo and 2 or 1,
        )  # default to webrtc framerate

    def create_video_track(self):
        return aiortc.VideoStreamTrack()

    def add_track_subscriber(
        self, track_id: str, callback: Optional[Callable[[PcmData], None]] = None
    ) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        """Subscribe to a track and optionally receive PCM audio data via callback.

        Args:
            track_id: The ID of the track to subscribe to.
            callback: Optional callback function that receives PcmData for audio tracks.
                     For GetStream, audio data is delivered through the connection's
                     'audio' event rather than this callback. This parameter is accepted
                     for interface compatibility but not currently used.

        Returns:
            The MediaStreamTrack for the subscribed track, or None if not found.
        """
        # Note: GetStream delivers audio through connection.on("audio") event handler
        # set up in join(), so the callback parameter is accepted but not used here.
        # The audio event handler already converts StreamPcmData to core PcmData.
        return self._connection.subscriber_pc.add_track_subscriber(track_id)

    async def publish_tracks(self, room, audio_track=None, video_track=None):  # type: ignore[override]
        """
        Add the tracks to publish audio and video.

        Args:
            room: Either the Room object (new calling convention) or audio_track (legacy).
            audio_track: The audio track to publish (new) or video_track (legacy).
            video_track: The video track to publish (new) or None (legacy).

        Note:
            This method supports both calling conventions:
            - Legacy: publish_tracks(audio_track, video_track)
            - New: publish_tracks(room, audio_track, video_track)
            The type: ignore comment is needed because we support both conventions.
        """
        # Handle both calling conventions
        if audio_track is None and video_track is None:
            # Legacy calling convention: publish_tracks(audio, video)
            # room parameter actually contains audio_track
            actual_audio = room
            actual_video = None
        elif video_track is None:
            # Legacy calling convention: publish_tracks(audio, video)
            # room contains audio, audio_track contains video
            actual_audio = room
            actual_video = audio_track
        else:
            # New calling convention: publish_tracks(room, audio, video)
            actual_audio = audio_track
            actual_video = video_track

        await self._connection.add_tracks(audio=actual_audio, video=actual_video)
        if actual_audio:
            logger.info("🤖 Agent ready to speak")
        if actual_video:
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
        # Note: Not calling super().close() as it's an abstract method with trivial body
        pass

    async def create_call(self, call_type: str, call_id: str) -> Call:
        """Create a call/room with the given type and ID.

        This method creates a new GetStream Call instance that can be used
        for joining and managing RTC sessions.

        Args:
            call_type: The type of call (e.g., "default", "video", "audio")
            call_id: Unique identifier for the call

        Returns:
            A Call instance representing the created call
        """
        call = self.client.video.call(call_type, call_id)
        await call.get_or_create(data={"created_by_id": self.agent_user_id})
        return call

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
                        is_global_banned=False,
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
