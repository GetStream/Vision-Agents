import logging
import asyncio
import os
import webbrowser
from typing import Optional, TYPE_CHECKING
from urllib.parse import urlencode
from uuid import uuid4

import aiortc
from getstream import AsyncStream
from getstream.chat.async_client import ChatClient
from getstream.models import ChannelInput
from getstream.video import rtc
from getstream.chat.async_channel import Channel
from getstream.video.async_call import Call
from getstream.video.rtc import ConnectionManager, audio_track
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant, TrackType
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

from vision_agents.core.edge import EdgeTransport, sfu_events
from vision_agents.plugins.getstream.stream_conversation import StreamConversation
from vision_agents.core.edge.types import Connection, User
from vision_agents.core.events.manager import EventManager
from vision_agents.core.edge import events
from vision_agents.core.utils import get_vision_agents_version

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent


class StreamConnection(Connection):
    def __init__(self, connection: ConnectionManager):
        super().__init__()
        # store the native connection object
        self._connection = connection

    async def close(self):
        await self._connection.leave()

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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.events = EventManager()
        self.events.register_events_from_module(events)
        self.events.register_events_from_module(sfu_events)
        self.channel: Optional[Channel] = None
        self.conversation: Optional[StreamConversation] = None
        self.channel_type = "videocall"
        # Track mapping: (user_id, session_id, track_type_int) -> {"track_id": str, "published": bool}
        # track_type_int is from TrackType enum (e.g., TrackType.TRACK_TYPE_AUDIO)
        self._track_map: dict = {}
        # Temporary storage for tracks before SFU confirms their type
        # track_id -> (user_id, session_id, webrtc_type_string)
        self._pending_tracks: dict = {}
        
        # Register event handlers
        self.events.subscribe(self._on_track_published)
        self.events.subscribe(self._on_track_removed)

    def _get_webrtc_kind(self, track_type_int: int) -> str:
        """Get the expected WebRTC kind (audio/video) for a SFU track type."""
        # Map SFU track types to WebRTC kinds
        if track_type_int in (TrackType.TRACK_TYPE_AUDIO, TrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO):
            return "audio"
        elif track_type_int in (TrackType.TRACK_TYPE_VIDEO, TrackType.TRACK_TYPE_SCREEN_SHARE):
            return "video"
        else:
            # Default to video for unknown types
            return "video"

    async def _on_track_published(self, event: sfu_events.TrackPublishedEvent):
        """Handle track published events from SFU - spawn TrackAddedEvent with correct type."""
        if not event.participant:
            return
        
        user_id = event.user_id
        session_id = event.payload.session_id
        track_type_int = event.payload.type  # TrackType enum int from SFU
        track_type_name = TrackType.Name(track_type_int)        
        # Get expected WebRTC kind for this track type
        expected_kind = self._get_webrtc_kind(track_type_int)
        track_key = (user_id, session_id, track_type_int)

        # First check if track already exists in map (e.g., from previous unpublish/republish)
        if track_key in self._track_map:
            self._track_map[track_key]["published"] = True
            self.logger.debug(f"Track marked as published (already existed): {track_key}")
            return

        # Wait for pending track to be populated (with 10 second timeout)
        # SFU might send TrackPublishedEvent before WebRTC processes track_added
        track_id = None
        timeout = 10.0
        poll_interval = 0.01  # 50ms
        elapsed = 0.0
        
        while elapsed < timeout:
            # Find pending track for this user/session with matching kind
            for tid, (pending_user, pending_session, pending_kind) in list(self._pending_tracks.items()):
                if (pending_user == user_id and 
                    pending_session == session_id and 
                    pending_kind == expected_kind):
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
            self.logger.info(f"Track published: {track_type_name} from {user_id}, track_id: {track_id} (waited {elapsed:.2f}s)")
            
            # NOW spawn TrackAddedEvent with correct type
            self.events.send(events.TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type_name,
                user=event.participant,
                user_metadata=event.participant
            ))
        else:
            raise TimeoutError(
                f"Timeout waiting for pending track: {track_type_name} ({expected_kind}) from user {user_id}, "
                f"session {session_id}. Waited {timeout}s but WebRTC track_added with matching kind was never received."
            )
    
    async def _on_track_removed(self, event: sfu_events.ParticipantLeftEvent | sfu_events.TrackUnpublishedEvent):
        """Handle track unpublished and participant left events."""
        if not event.participant:
            return
        
        # Extract fields based on event type
        participant = event.participant
        user_id = participant.user_id
        session_id = participant.session_id
        
        # Determine which tracks to remove
        if hasattr(event, 'payload') and hasattr(event.payload, 'type'):
            # TrackUnpublishedEvent - single track
            tracks_to_remove = [event.payload.type]
            event_desc = "Track unpublished"
        else:
            # ParticipantLeftEvent - all published tracks
            tracks_to_remove = participant.published_tracks
            event_desc = "Participant left"
        
        track_names = [TrackType.Name(t) for t in tracks_to_remove]
        self.logger.info(f"{event_desc}: {user_id}, tracks: {track_names}")
        
        # Mark each track as unpublished and send TrackRemovedEvent
        for track_type_int in tracks_to_remove:
            track_type_name = TrackType.Name(track_type_int)
            track_key = (user_id, session_id, track_type_int)
            track_info = self._track_map.get(track_key)
            
            if track_info:
                track_id = track_info["track_id"]
                self.events.send(events.TrackRemovedEvent(
                    plugin_name="getstream",
                    track_id=track_id,
                    track_type=track_type_name,
                    user=participant,
                    user_metadata=participant
                ))
                # Mark as unpublished instead of removing
                self._track_map[track_key]["published"] = False
            else:
                self.logger.warning(f"Track not found in map: {track_key}")

    async def create_conversation(self, call: Call, user, instructions):
        chat_client: ChatClient = call.client.stream.chat
        self.channel = await chat_client.get_or_create_channel(
            self.channel_type,
            call.id,
            data=ChannelInput(created_by_id=user.id),
        )
        self.conversation = StreamConversation(
            instructions, [], self.channel.data.channel, chat_client
        )
        return self.conversation

    async def create_user(self, user: User):
        return await self.client.create_user(name=user.name, id=user.id)

    async def join(self, agent: "Agent", call: Call) -> StreamConnection:
        """
        The logic for joining a call is different for each edge network/realtime audio/video provider

        This function
        - initializes the chat channel
        - has the agent.agent_user join the call
        - connect incoming audio/video to the agent
        - connecting agent's outgoing audio/video to the call

        TODO:
        - process track flow

        """
        # Traditional mode - use WebRTC connection
        # Configure subscription for audio and video
        subscription_config = SubscriptionConfig(
            default=self._get_subscription_config()
        )

        try:
            # Open RTC connection and keep it alive for the duration of the returned context manager
            connection = await rtc.join(
                call, agent.agent_user.id, subscription_config=subscription_config
            )
            await connection.__aenter__() # TODO: weird API? there should be a manual version
        except Exception:
            raise

        self._connection = connection

        original_on_subscriber_offer = self._connection._on_subscriber_offer  # type: ignore[attr-defined]

        async def _safe_on_subscriber_offer(event):
            if self._connection.subscriber_pc is None:  # type: ignore[attr-defined]
                self.logger.debug("Ignoring subscriber offer after subscriber_pc teardown")
                return
            await original_on_subscriber_offer(event)

        self._connection._on_subscriber_offer = _safe_on_subscriber_offer  # type: ignore[attr-defined]

        @self._connection.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            self.events.send(events.AudioReceivedEvent(
                plugin_name="getstream",
                pcm_data=pcm,
                participant=participant,
                user_metadata=participant
            ))

        self.events.silent(events.AudioReceivedEvent)

        @self._connection.on("track_added")
        async def on_track(track_id, track_type, user):
            # Store track in pending map - wait for SFU to confirm type before spawning TrackAddedEvent
            self._pending_tracks[track_id] = (user.user_id, user.session_id, track_type)
            self.logger.info(f"Track received from WebRTC (pending SFU confirmation): {track_id}, type: {track_type}, user: {user.user_id}")


        standardize_connection = StreamConnection(connection)

        return standardize_connection

    def create_audio_track(self, framerate: int = 48000, stereo: bool = True):
        return audio_track.AudioStreamTrack(framerate=framerate, stereo=stereo) # default to webrtc framerate

    def create_video_track(self):
        return aiortc.VideoStreamTrack()

    def add_track_subscriber(self, track_id: str) -> Optional[aiortc.mediastreams.MediaStreamTrack]:
        return self._connection.subscriber_pc.add_track_subscriber(track_id)

    async def publish_tracks(self, audio_track, video_track):
        """
        Add the tracks to publish audio and video
        """
        await self._connection.add_tracks(audio=audio_track, video=video_track)
        if audio_track:
            self.logger.info("🤖 Agent ready to speak")
        if video_track:
            self.logger.info("🎥 Agent ready to publish video")
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

    def close(self):
        # Note: Not calling super().close() as it's an abstract method with trivial body
        pass

    async def open_demo(self, call: Call) -> str:
        client = call.client.stream

        # Create a human user for testing
        human_id = f"user-{uuid4()}"
        name = "Human User"

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
            "video_encoder": "vp8",
            "bitrate": 12000000,
            "w": 1920,
            "h": 1080,
            # TODO: FPS..., aim at 60fps
        }

        url = f"{base_url}{call.id}?{urlencode(params)}"
        print(f"🌐 Opening browser to: {url}")

        try:
            webbrowser.open(url)
            print("✅ Browser opened successfully!")
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
            print(f"Please manually open this URL: {url}")

        return url

    def open_pronto(self, api_key: str, token: str, call_id: str):
        """Open browser with the video call URL."""
        # Use the same URL pattern as the working workout assistant example
        base_url = (
            f"{os.getenv('EXAMPLE_BASE_URL', 'https://pronto-staging.getstream.io')}/join/"
        )
        params = {
            "api_key": api_key,
            "token": token,
            "skip_lobby": "true",
            "video_encoder": "vp8",
        }

        url = f"{base_url}{call_id}?{urlencode(params)}"
        self.logger.info(f"🌐 Opening browser: {url}")

        try:
            webbrowser.open(url)
            self.logger.info("✅ Browser opened successfully!")
        except Exception as e:
            self.logger.error(f"❌ Failed to open browser: {e}")
            self.logger.info(f"Please manually open this URL: {url}")
