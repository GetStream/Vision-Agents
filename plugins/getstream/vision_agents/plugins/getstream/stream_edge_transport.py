import logging
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
from getstream.video.rtc import audio_track, ConnectionManager
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType, Participant
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import TrackSubscriptionConfig, SubscriptionConfig

from vision_agents.plugins.getstream.stream_conversation import StreamConversation
from vision_agents.core.edge import EdgeTransport
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
        self.channel: Optional[Channel] = None
        self.conversation: Optional[StreamConversation] = None
        self.channel_type = "videocall"

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

        @self._connection.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            self.events.send(events.AudioReceivedEvent(
                plugin_name="getstream",
                pcm_data=pcm,
                participant=participant,
                user_metadata=participant
            ))

        @self._connection.on("track_added")
        async def on_track(track_id, track_type, user):
            # TODO: maybe make it easy to subscribe only to video tracks?
            self.events.send(events.TrackAddedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type,
                user=user,
                user_metadata=user
            ))

            _, track = self._connection.subscriber_pc.track_map[track_id]
            track.on("ended", lambda: self.events.send(events.TrackEndedEvent(
                plugin_name="getstream",
                track_id=track_id,
                track_type=track_type,
                user=user,
                user_metadata=user
            )))

        @self._connection.on("call_ended")
        async def call_ended(*args, **kwargs):
            self.events.send(events.CallEndedEvent(
                plugin_name="getstream",
                args=args,
                kwargs=kwargs
            ))

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
