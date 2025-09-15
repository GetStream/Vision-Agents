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
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType, Participant
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



