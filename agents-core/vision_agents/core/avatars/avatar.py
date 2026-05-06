import abc
import logging

import aiortc
from vision_agents.core.agents.inference import AudioOutputStream
from vision_agents.core.events import EventManager
from vision_agents.core.observability import MetricsCollector

logger = logging.getLogger(__name__)


class Avatar(abc.ABC):
    """Base class for avatar plugins (passthrough mode).

    Avatars consume the agent's audio output and produce a synced video
    and audio feed of a virtual character. They own the agent's outbound
    video and audio tracks; their output never feeds back into the LLM
    or video processors.

    Wiring:
        - The agent calls ``attach_audio_input(stream)`` during
          ``__init__`` so the avatar can decide how to consume the
          inference flow's audio output.
        - Subclasses expose the avatar's lipsynced PCM via
          ``output_audio_stream()``. The agent's audio producer drains
          that stream into the outbound rtc track.
        - The provider's video frames are exposed via
          ``output_video_track()`` and become the agent's outbound video.

    Lifecycle:
        - ``output_video_track()`` is queried during ``Agent.__init__``.
        - ``attach_audio_input(stream)`` is called during ``Agent.__init__``.
        - ``start()`` is called during ``Agent.join()`` to open the
          provider connection and begin consuming the input stream.
        - ``stop()`` is called during ``Agent.close()`` for teardown.
        - ``interrupt()`` may be called at any time to stop the in-flight
          utterance at the provider.
    """

    provider_name: str | None = None

    def __init__(self) -> None:
        self.events = EventManager()
        self.metrics = MetricsCollector()
        # Avatar's input is the Agent's output
        self.input_audio_stream: AudioOutputStream | None = None

    def attach_audio_input(self, stream: AudioOutputStream) -> None:
        """Receive the Agent's audio output stream.

        Called during ``Agent.__init__``.
        The avatar decides how and when
        to read from this stream — typically in ``start()`` it spins up a
        task that consumes items and forwards them to the provider.

        This method can be overridden by subclasses to customize
        how audio is fed into the avatar.
        """
        self.input_audio_stream = stream

    @abc.abstractmethod
    def video_output(self) -> aiortc.VideoStreamTrack:
        """Return the outbound video track this avatar publishes to the call."""

    @abc.abstractmethod
    def audio_output(self) -> AudioOutputStream:
        """Return the outbound audio stream this avatar publishes to the call."""

    @abc.abstractmethod
    async def start(self) -> None:
        """
        Start consuming the input stream.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Tear down the provider connection and cancel any consumer tasks."""
