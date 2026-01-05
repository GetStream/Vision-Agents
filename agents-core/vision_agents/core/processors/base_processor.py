import abc
import logging
import typing
from typing import Optional

import aiortc
from getstream.video.rtc import PcmData
from vision_agents.core.utils.video_forwarder import VideoForwarder

if typing.TYPE_CHECKING:
    from vision_agents.core import Agent

logger = logging.getLogger(__name__)


class Processor(abc.ABC):
    """
    A base class for all audio and video processors.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Processor name.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Close the processor and clean up resources when the application exits.
        """

    def attach_agent(self, agent: "Agent") -> None:
        """
        A method to perform action with an Agent inside the processor (e.g. register custom events).

        Args:
            agent: Agent instance.
        Returns: None
        """
        ...


class VideoPublisher(Processor, metaclass=abc.ABCMeta):
    """
    Base class for video processors that publish outgoing video tracks.
    Example: avatar plugin generating video on the fly.
    """

    @abc.abstractmethod
    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        """
        Returns a video track with the processed frames.
        """


class VideoProcessor(Processor, metaclass=abc.ABCMeta):
    """
    Base class for video processors that process incoming video tracks.
    Example: a plugin that logs video frames for analysis.
    """

    @abc.abstractmethod
    async def process_video(
        self,
        track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        A method to start processing a video track.
        It's called by the Agent every time a new track is published.
        """


class VideoProcessorPublisher(VideoProcessor, VideoPublisher, metaclass=abc.ABCMeta):
    """
    Base class for video processors that process incoming video tracks and publish the video
    back to the call.
    Example: object detection plugin that annotates video frames with detection results.
    """


class AudioPublisher(Processor, metaclass=abc.ABCMeta):
    """
    Base class for audio processors that publish outgoing audio tracks.
    """

    @abc.abstractmethod
    def publish_audio_track(self) -> aiortc.AudioStreamTrack: ...


class AudioProcessor(Processor, metaclass=abc.ABCMeta):
    """
    Base class for all audio processors that process incoming audio tracks.
    """

    @abc.abstractmethod
    async def process_audio(self, audio_data: PcmData) -> None:
        """Process audio data. Override this method to implement audio processing.

        Args:
            audio_data: PcmData containing audio samples and metadata.
                       The participant is stored in audio_data.participant.
        """


class AudioProcessorPublisher(AudioPublisher, AudioProcessor, metaclass=abc.ABCMeta):
    """
    Base class for audio processors that both process incoming audio tracks and publish the audio back to the call.
    """
