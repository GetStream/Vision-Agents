import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiortc
import numpy as np
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.events import AudioReceivedEvent, TrackAddedEvent
from vision_agents.core.edge.types import Connection, Participant, TrackType, User

from .devices import AudioInputDevice, AudioOutputDevice, CameraDevice
from .tracks import LocalOutputAudioTrack, LocalVideoTrack

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

PLUGIN_NAME = "local"
LOCAL_VIDEO_TRACK_ID = "local-video-track"

logger = logging.getLogger(__name__)


@dataclass
class LocalCall:
    """Minimal Call-compatible object for local transport."""

    id: str


# TODO: How to make screencapture work?
# TODO: Use Tkinter to display video output?


class LocalEdge(EdgeTransport):
    """EdgeTransport implementation for local audio/video I/O.

    Uses AudioInputDevice / AudioOutputDevice for microphone capture and
    speaker playback (defaults to sounddevice). Subclass these device
    classes to use alternative hardware (e.g. GStreamer).
    """

    def __init__(
        self,
        audio_input: AudioInputDevice,
        audio_output: AudioOutputDevice,
        video_input: CameraDevice | None = None,
        video_width: int = 640,
        video_height: int = 480,
        video_fps: int = 30,
    ):
        super().__init__()

        self._audio_input = audio_input
        self._audio_output = audio_output

        self._video_input = video_input.device if video_input else None
        self._video_width = video_width
        self._video_height = video_height
        self._video_fps = video_fps

        self._participant = Participant(
            original=None,
            user_id="local",
            id="local",
        )

        self._mic_task: asyncio.Task[None] | None = None
        self._running = False
        self._audio_track: LocalOutputAudioTrack | None = None
        self._video_track: LocalVideoTrack | None = None
        self._connection: LocalConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def publish_tracks(
        self,
        audio_track: aiortc.MediaStreamTrack | None,
        video_track: aiortc.MediaStreamTrack | None,
    ) -> None:
        """Publish the agent's media tracks locally."""
        if audio_track is not None and isinstance(audio_track, LocalOutputAudioTrack):
            audio_track.start()
            logger.info("Audio track published and started")

        if video_track is not None and isinstance(video_track, LocalVideoTrack):
            logger.info("Video track published")

    def create_audio_track(
        self, sample_rate: int = 48000, stereo: bool = True
    ) -> "LocalOutputAudioTrack":
        """Create an audio track that plays through the audio output backend."""
        self._audio_track = LocalOutputAudioTrack(
            audio_output=self._audio_output,
        )
        return self._audio_track

    def create_video_track(self) -> LocalVideoTrack | None:
        """Create a video track for the agent's camera input."""
        if self._video_input is None:
            logger.debug("No video device configured, skipping video track creation")
            return None

        self._video_track = LocalVideoTrack(
            device=self._video_input,
            width=self._video_width,
            height=self._video_height,
            fps=self._video_fps,
        )
        return self._video_track

    def add_track_subscriber(self, track_id: str) -> LocalVideoTrack | None:
        """Return the local camera video track if available."""
        if track_id == LOCAL_VIDEO_TRACK_ID and self._video_track is not None:
            return self._video_track
        return None

    async def join(
        self, agent: "Agent", call: Any = None, **kwargs: Any
    ) -> "LocalConnection":
        """Start microphone capture and optionally camera."""
        await self._start_audio()

        if self._video_input is not None:
            video_track = self.create_video_track()
            if video_track is not None:
                self.events.send(
                    TrackAddedEvent(
                        plugin_name=PLUGIN_NAME,
                        track_id=LOCAL_VIDEO_TRACK_ID,
                        track_type=TrackType.VIDEO,
                        participant=self._participant,
                    )
                )
                logger.info("Camera video track added")

        self._connection = LocalConnection(self)
        return self._connection

    async def close(self) -> None:
        """Stop audio/video and release all resources."""
        await self._stop_audio()
        self._connection = None

    async def authenticate(self, user: User) -> None:
        # Local transport does not require any auth
        return

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        """Not supported for local transport."""
        logger.warning(
            "LocalEdge does not have a demo UI. "
            "Audio is captured from the microphone and played on the speakers."
        )

    async def create_call(self, call_id: str, **kwargs: Any) -> LocalCall:
        return LocalCall(id=call_id)

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        raise NotImplementedError("LocalEdge does not support send_custom_event")

    async def create_conversation(
        self, call: Any, user: User, instructions: str
    ) -> None:
        #
        return None

    def _emit_audio_event(self, data: np.ndarray) -> None:
        """Convert raw numpy audio to PcmData and emit AudioReceivedEvent."""
        samples = data.flatten().astype(np.int16)
        pcm = PcmData(
            samples=samples,
            sample_rate=self._audio_input.sample_rate,
            format=AudioFormat.S16,
            channels=self._audio_input.channels,
        )
        pcm.participant = self._participant

        self.events.send(
            AudioReceivedEvent(
                plugin_name=PLUGIN_NAME,
                pcm_data=pcm,
                participant=self._participant,
            )
        )

    async def _mic_polling_loop(self) -> None:
        """Read from the mic on a background thread and dispatch events to the event loop."""
        assert self._loop is not None
        loop = self._loop

        def _read_loop() -> None:
            while self._running:
                data = self._audio_input.read()
                if data is not None:
                    loop.call_soon_threadsafe(self._emit_audio_event, data)

        try:
            await asyncio.to_thread(_read_loop)
        except asyncio.CancelledError:
            logger.debug("Mic polling loop cancelled")
            raise

    async def _start_audio(self) -> None:
        """Start microphone capture via the audio input backend."""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        self._audio_input.start()
        logger.info(
            "Started microphone: %dHz, %d channels",
            self._audio_input.sample_rate,
            self._audio_input.channels,
        )
        self._mic_task = asyncio.create_task(self._mic_polling_loop())

    async def _stop_audio(self) -> None:
        """Stop all audio and video streams."""
        self._running = False

        if self._mic_task is not None:
            self._mic_task.cancel()
            try:
                await self._mic_task
            except asyncio.CancelledError:
                pass
            self._mic_task = None

        self._audio_input.stop()
        logger.info("Stopped microphone")

        if self._audio_track is not None:
            self._audio_track.stop()

        if self._video_track is not None:
            self._video_track.stop()
            self._video_track = None


class LocalConnection(Connection):
    """Connection wrapper for local transport."""

    def __init__(self, transport: "LocalEdge"):
        super().__init__()
        self._transport = transport

    def idle_since(self) -> float:
        """Local transport is never idle."""
        return 0.0

    async def wait_for_participant(self, timeout: float | None = None) -> None:
        """Local user is always present, return immediately."""
        return

    async def close(self, timeout: float = 2.0) -> None:
        """Close the local connection."""
        await self._transport.close()
