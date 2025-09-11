"""
Abstraction for stream vs other services here
"""
import abc
import logging
import os
import webbrowser
from urllib.parse import urlencode
from uuid import uuid4

from getstream import Stream
from getstream.models import UserRequest, Call
from typing import TYPE_CHECKING

if TYPE_CHECKING:

    from stream_agents.core.agents import Agent


class EdgeTransport(abc.ABC):
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
        self.client = Stream.from_env()
        self.logger = logging.getLogger(self.__class__.__name__)

    def join(self, agent: "Agent", call: Call):
        """
        The logic for joining a call is different for each edge network/realtime audio/video provider

        This function
        - initializes the chat channel
        - has the agent.agent_user join the call
        - connect incoming audio/video to the agent
        - connecting agent's outgoing audio/video to the call

        TODO:
        - not implemented yet since this will change after realtime STS integration
        """
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

