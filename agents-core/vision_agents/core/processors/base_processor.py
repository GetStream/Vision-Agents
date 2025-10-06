import abc
import asyncio
import logging
from pathlib import Path
from typing import Protocol, Any, List, Optional
from enum import Enum

import aiortc
from PIL import Image
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

"""
TODO:
- simple audio test
- simple video test
- properly forward to image processors from agent (easy)
- figure out aysncio flow for video track recv() loop

"""


class ProcessorType(Enum):
    """Enum for different processor types based on mixins."""

    AUDIO = "process_audio"
    VIDEO = "process_video"
    IMAGE = "process_image"
    VIDEO_PUBLISHER = "publish_video_track"
    AUDIO_PUBLISHER = "publish_audio_track"


class Processor(Protocol):
    def state(self) -> Any:
        pass

    def input(self) -> Any:
        pass

    def close(self):
        pass


class IntervalProcessor(Processor):
    # TODO: add interval loop
    pass


class AudioProcessorMixin(abc.ABC):
    @abc.abstractmethod
    async def process_audio(
        self, audio_data: bytes, participant: models_pb2.Participant
    ) -> None:
        """Process audio data. Override this method to implement audio processing."""
        pass


class VideoProcessorMixin(abc.ABC):
    @abc.abstractmethod
    async def process_video(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        participant: models_pb2.Participant,
    ):
        pass


class ImageProcessorMixin(abc.ABC):
    @abc.abstractmethod
    async def process_image(
        self, image: Image.Image, participant: models_pb2.Participant
    ):
        pass


class VideoPublisherMixin:
    def publish_video_track(self):
        return aiortc.VideoStreamTrack()


class AudioPublisherMixin:
    def publish_audio_track(self):
        return aiortc.AudioStreamTrack()


def filter_processors(
    processors: List[Processor], processor_type: ProcessorType
) -> List[Processor]:
    """
    Filter processors based on the processor type using hasattr checks.

    Args:
        processors: List of processor instances to filter
        processor_type: ProcessorType enum value to filter by

    Returns:
        List of processors that have the required method/attribute for the given type
    """
    filtered = []
    method_name = processor_type.value

    for processor in processors:
        if hasattr(processor, method_name):
            filtered.append(processor)

    return filtered


class AudioVideoProcessor(Processor):
    def __init__(
        self,
        interval: int = 3,
        receive_audio: bool = False,
        receive_video: bool = True,
        *args,
        **kwargs,
    ):
        self.interval = interval
        self.last_process_time = 0

        if hasattr(self, "create_audio_track"):
            self.audio_track = self.create_audio_track()

        if hasattr(self, "create_video_track"):
            self.video_track = self.create_video_track()

    def state(self):
        # Returns relevant data for the conversation with the LLM
        pass

    def should_process(self) -> bool:
        """Check if it's time to process based on the interval."""
        current_time = asyncio.get_event_loop().time()

        if current_time - self.last_process_time >= self.interval:
            self.last_process_time = int(current_time)
            return True
        return False


class AudioLogger(AudioVideoProcessor, AudioProcessorMixin):
    def __init__(self, interval: int = 2):
        super().__init__(interval, receive_audio=True, receive_video=False)
        self.audio_count = 0
        self.last_audio_time = 0

    async def process_audio(
        self, audio_data: bytes, user_id: str, metadata: Optional[dict[Any, Any]] = None
    ) -> None:
        """Log audio data information."""
        asyncio.get_event_loop().time()

        if self.should_process():
            self.audio_count += 1

            logging.info(
                f"🎵 Audio #{self.audio_count} from {user_id}: {len(audio_data)} bytes"
            )


class ImageCapture(AudioVideoProcessor, ImageProcessorMixin):
    """Handles video frame capture and storage at regular intervals."""

    def __init__(
        self, output_dir: str = "captured_frames", interval: int = 3, *args, **kwargs
    ):
        super().__init__(interval=interval, receive_audio=False, receive_video=True)
        self.output_dir = Path(output_dir)
        self.frame_count = 0

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f"📁 Saving captured frames to: {self.output_dir.absolute()}")

    async def process_image(
        self,
        image: Image.Image,
        user_id: str,
        metadata: Optional[dict[Any, Any]] = None,
    ):
        # Check if enough time has passed since last capture
        if not self.should_process():
            return None

        logging.info(f"📸 ImageCapture running process_image for user {user_id}")

        timestamp = int(asyncio.get_event_loop().time())
        filename = f"frame_{user_id}_{timestamp}_{self.frame_count:04d}.jpg"
        filepath = self.output_dir / filename

        # Save the frame as JPG
        image.save(filepath, "JPEG", quality=90)

        self.frame_count += 1
        logging.info(
            f"📸 Captured frame {self.frame_count}: {filename} ({image.width}x{image.height})"
        )

        return str(filepath)
