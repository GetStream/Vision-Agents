"""
Abstraction for stream vs other services here
"""
import abc
import logging
import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

import aiortc
from getstream import Stream
from getstream.models import UserRequest, Call
from typing import TYPE_CHECKING

from getstream.video import rtc
from getstream.video.rtc import audio_track
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType, Participant, VideoDimension
from getstream.video.rtc.track_util import PcmData
from getstream.video.rtc.tracks import TrackSubscriptionConfig, SubscriptionConfig
from pyee.asyncio import AsyncIOEventEmitter

if TYPE_CHECKING:

    from stream_agents.core.agents import Agent


class EdgeTransport(AsyncIOEventEmitter, abc.ABC):
    """
    To normalize

    - join method
    - call/room object
    - open demo/ browser
    """

    def open_demo(self, *args, **kwargs):
        pass


class StreamEdge(EdgeTransport):
    """
    StreamEdge uses getstream.io's edge network. To support multiple vendors, this means we expose

    """
    client: Stream

    def __init__(self, **kwargs):
        # Initialize Stream client
        super().__init__()
        self.client = Stream.from_env()
        self.logger = logging.getLogger(self.__class__.__name__)

    async def join(self, agent: "Agent", call: Call):
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

        # Open RTC connection and keep it alive for the duration of the returned context manager
        connection_cm = await rtc.join(
            call, agent.agent_user.id, subscription_config=subscription_config
        )
        connection = await connection_cm.__aenter__()
        self._connection = connection

        @self._connection.on("audio")
        async def on_audio_received(pcm: PcmData, participant: Participant):
            self.emit("audio", pcm, participant)

        @self._connection.on("track_added")
        async def on_track(track_id, track_type, user):
            # TODO: maybe make it easy to subscribe only to video tracks?
            self.emit("track_added", track_id, track_type, user)

        # Some SDK versions emit 'track_published' instead of 'track_added'.
        # Forward it to the same 'track_added' hook for compatibility.
        @self._connection.on("track_published")
        async def on_track_published(*args, **kwargs):
            try:
                self.emit("track_added", *args, **kwargs)
            except Exception:
                pass

        return connection_cm

    def create_audio_track(self):
        return audio_track.AudioStreamTrack(framerate=16000)

    def create_video_track(self):
        return aiortc.VideoStreamTrack()

    async def publish_tracks(self, audio_track, video_track):
        """
        Add the tracks to publish audio and video
        """
        await self._connection.add_tracks(audio=audio_track, video=video_track)
        if audio_track:
            self.logger.debug("🤖 Agent ready to speak")
        if video_track:
            self.logger.debug("🎥 Agent ready to publish video")
        # In Realtime mode we directly publish the provider's output track; no extra forwarding needed

    def _get_subscription_config(self):
        # Request 720p video and subscribe to both video and audio tracks
        return TrackSubscriptionConfig(
            track_types=[
                TrackType.TRACK_TYPE_VIDEO,
                TrackType.TRACK_TYPE_AUDIO,
            ],
            video_dimension=VideoDimension(width=1280, height=720),
        )

    def close(self):
        pass

    def open_demo(self, call: Call) -> str:
        client = call.client.stream

        # Create a human user for testing
        human_id = f"user-{uuid4()}"

        client.upsert_users(UserRequest(id=human_id, name="Human User"))

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
            "video_encoder": "vp8",
            "bitrate": 2000000,
            "w": 1920,
            "h": 1080,
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

