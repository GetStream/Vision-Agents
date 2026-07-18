import asyncio
import contextlib
import logging
import os

from anam import (
    AgentAudioInputConfig,
    AgentAudioInputStream,
    AnamClient,
    AnamEvent,
    ClientOptions,
    PersonaConfig,
    Session,
)
from getstream.video.rtc.track_util import FrameResampler, PcmData
from vision_agents.core.agents.inference import (
    AudioOutputChunk,
    AudioOutputFlush,
    AudioOutputStream,
)
from vision_agents.core.avatars import Avatar
from vision_agents.core.avatars.av_synchronizer import AVSynchronizer
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)

# Sample rate Anam expects for agent audio input.
AVATAR_SAMPLE_RATE = 24000

CLOSE_TIMEOUT = 5.0


def _task_done_callback(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "Background task %s failed", task.get_name(), exc_info=task.exception()
        )


class AnamAvatar(Avatar):
    """Anam avatar plugin.

    References:
    - https://anam.ai/
    - https://github.com/anam-org/python-sdk

    Sends TTS audio to Anam and receives synchronized
    avatar video and audio back.
    """

    provider_name = "anam_avatar"

    def __init__(
        self,
        avatar_id: str | None = None,
        api_key: str | None = None,
        client_options: ClientOptions | None = None,
        connect_timeout: float | None = None,
        session_ready_timeout: float | None = None,
        width: int = 720,
        height: int = 480,
        fps: int = 30,
        buffer_seconds: float = 1.0,
    ):
        """Initialize the Anam avatar publisher.

        Args:
            avatar_id: Anam avatar ID. Uses ANAM_AVATAR_ID env var if not provided.
            api_key: Anam API key. Uses ANAM_API_KEY env var if not provided.
            client_options: Optional Anam client configuration options.
            connect_timeout: Seconds to wait for the connection to be established.
                None means wait indefinitely.
            session_ready_timeout: Seconds to wait for the session to become ready.
                None means wait indefinitely.
            width: Output video width in pixels.
            height: Output video height in pixels.
            fps: Output video frame rate. Must be > 0.
            buffer_seconds: Max video buffer depth in seconds. Caps how many frames
                can be queued ahead of audio playback. Must be > 0.
        """
        super().__init__()
        api_key = api_key or os.getenv("ANAM_API_KEY")
        if not api_key:
            raise ValueError("Anam API key not provided")
        avatar_id = avatar_id or os.getenv("ANAM_AVATAR_ID")
        if not avatar_id:
            raise ValueError("Anam avatar ID not provided")
        if buffer_seconds <= 0:
            raise ValueError("buffer_seconds must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")

        self._client = AnamClient(
            api_key=api_key,
            persona_config=PersonaConfig(
                avatar_id=avatar_id,
                enable_audio_passthrough=True,
            ),
            options=client_options,
        )
        # Subscribe to Anam client events
        self._client.on(AnamEvent.CONNECTION_ESTABLISHED)(
            self._on_connection_established
        )
        self._client.on(AnamEvent.CONNECTION_CLOSED)(self._on_connection_closed)
        self._client.on(AnamEvent.SESSION_READY)(self._on_session_ready)

        self._sync = AVSynchronizer(
            width=width,
            height=height,
            fps=fps,
            max_queue_size=int(fps * buffer_seconds),
        )

        self._connect_timeout = connect_timeout
        self._session_ready_timeout = session_ready_timeout

        self._connected = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._exit_stack = contextlib.AsyncExitStack()
        self._real_session: Session | None = None
        self._audio_input_stream: AgentAudioInputStream | None = None
        self._resampler = FrameResampler(
            rate=AVATAR_SAMPLE_RATE, layout="mono", format="s16", frame_size=0
        )
        self._audio_receiver_task: asyncio.Task[None] | None = None
        self._video_receiver_task: asyncio.Task[None] | None = None
        self._audio_input_task: asyncio.Task[None] | None = None

    def video_output(self) -> QueuedVideoTrack:
        """Return the video track that receives avatar video frames."""
        return self._sync.video_output

    def audio_output(self) -> AudioOutputStream:
        """Return the video track that receives avatar video frames."""
        return self._sync.audio_output

    async def start(self) -> None:
        """Connect to Anam. Called by Agent via _apply("start") during join()."""
        await self._connect()

    async def close(self) -> None:
        """
        Close the Anam avatar publisher, cancel audio & video processing tasks
        and release resources.
        """

        for task in (
            self._audio_input_task,
            self._audio_receiver_task,
            self._video_receiver_task,
        ):
            if task is not None:
                await cancel_and_wait(task)

        self._sync.close()
        try:
            # aiortc/websocket teardown blocks forever when the call is already
            # gone (no peer to ack the DTLS/ICE and WS close), so bound it.
            await asyncio.wait_for(self._exit_stack.aclose(), timeout=CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Timed out closing Anam session")
        except Exception:
            logger.warning("Failed to close Anam avatar publisher", exc_info=True)
        finally:
            # Close the client even if the session teardown above timed out or
            # failed, so its HTTP/WS resources don't leak in that scenario.
            with contextlib.suppress(Exception):
                await self._client.close()
            logger.debug("Anam avatar publisher closed")

    @property
    def _session(self) -> Session:
        if self._real_session is None:
            raise RuntimeError("Anam avatar session not initialized")
        return self._real_session

    async def _process_audio_input(self) -> None:
        """
        Process audio input from the Agent
        """

        # Init the avatar's input stream early
        self._init_avatar_input_stream()
        async for item in self.input_audio_stream:
            if isinstance(item, AudioOutputChunk):
                # Received normal audio, send it to the avatar.
                if item.data is not None:
                    await self._send_audio(item.data)
                # Received final audio chunk (end-of-utterance), flush avatar's audio
                if item.final:
                    await self._end_turn()

            elif isinstance(item, AudioOutputFlush):
                # Audio was interrupted: discard the resampler tail so it doesn't
                # bleed into the next turn.
                self._resampler.flush()
                await self._end_turn()
                await self._sync.flush()
                await self._session.interrupt()

    async def _video_receiver(self) -> None:
        """
        Receive video from avatar.
        """

        async for frame in self._session.video_frames():
            try:
                await self._sync.write_video(frame)
            except Exception:
                logger.warning("Failed to write video frame", exc_info=True)

    async def _audio_receiver(self) -> None:
        """
        Receive audio from avatar.
        """

        async for frame in self._session.audio_frames():
            try:
                await self._sync.write_audio(PcmData.from_av_frame(frame))
            except Exception:
                logger.warning("Failed to send audio frame", exc_info=True)

    def _init_avatar_input_stream(self) -> AgentAudioInputStream:
        if self._audio_input_stream is None:
            self._audio_input_stream = self._session.create_agent_audio_input_stream(
                AgentAudioInputConfig(
                    encoding="pcm_s16le",
                    sample_rate=AVATAR_SAMPLE_RATE,
                    channels=1,
                )
            )
        return self._audio_input_stream

    async def _send_audio(self, pcm: PcmData) -> None:
        """
        Resample agent audio to the avatar's rate and send it.
        """
        stream = self._init_avatar_input_stream()
        for frame in self._resampler.resample(pcm):
            await stream.send_audio_chunk(frame.to_ndarray().tobytes())

    async def _end_turn(self) -> None:
        """
        Drain the resampler tail and signal end of the turn to the avatar.
        """
        if self._audio_input_stream is None:
            return
        # Flush the resampler tail so the last partial frame isn't dropped.
        for frame in self._resampler.flush():
            await self._audio_input_stream.send_audio_chunk(
                frame.to_ndarray().tobytes()
            )
        await self._audio_input_stream.end_sequence()

    async def _connect(self) -> None:
        if self._real_session is None:
            self._real_session = await self._exit_stack.enter_async_context(
                self._client.connect()
            )

        await self._wait_connected()
        await self._wait_session_ready()
        if self._audio_receiver_task is None:
            self._audio_receiver_task = asyncio.create_task(self._audio_receiver())
            self._audio_receiver_task.add_done_callback(_task_done_callback)
        if self._video_receiver_task is None:
            self._video_receiver_task = asyncio.create_task(self._video_receiver())
            self._video_receiver_task.add_done_callback(_task_done_callback)
        if self._audio_input_task is None:
            self._audio_input_task = asyncio.create_task(self._process_audio_input())
            self._audio_input_task.add_done_callback(_task_done_callback)

    async def _on_connection_established(self):
        """
        Called when the Anam connection is established.
        """
        self._connected.set()
        logger.debug("Anam connection established")

    async def _on_connection_closed(self, code: str, reason: str | None):
        """
        Called when the Anam connection is closed.
        """
        if reason:
            logger.warning(f"Closing Anam connection: {code} - {reason}")
        else:
            logger.debug("Closing Anam connection")

    async def _on_session_ready(self) -> None:
        """
        Called when the Anam session is ready to receive audio.
        Audio sent before the session is ready will be dropped.
        """
        self._session_ready.set()

    async def _wait_connected(self) -> None:
        try:
            await asyncio.wait_for(
                self._connected.wait(), timeout=self._connect_timeout
            )
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for Anam connection to be established")
            raise
        finally:
            self._connected.clear()

    async def _wait_session_ready(self) -> None:
        try:
            await asyncio.wait_for(
                self._session_ready.wait(), timeout=self._session_ready_timeout
            )
        except asyncio.TimeoutError:
            logger.error("Timed out waiting for Anam session to get ready")
            raise
        finally:
            self._session_ready.clear()
