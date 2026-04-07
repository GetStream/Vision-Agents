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
from getstream.video.rtc.audio_track import AudioStreamTrack
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import Agent
from vision_agents.core.llm.events import (
    RealtimeAudioOutputDoneEvent,
    RealtimeAudioOutputEvent,
)
from vision_agents.core.llm.realtime import Realtime
from vision_agents.core.processors.base_processor import AudioPublisher, VideoPublisher
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.turn_detection import TurnStartedEvent
from vision_agents.core.utils.av_synchronizer import AVSynchronizer
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.utils.video_track import QueuedVideoTrack

logger = logging.getLogger(__name__)


def _task_done_callback(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "Background task %s failed", task.get_name(), exc_info=task.exception()
        )


class AnamAvatarPublisher(AudioPublisher, VideoPublisher):
    """Anam avatar video and audio publisher.

    References:
    - https://anam.ai/
    - https://github.com/anam-org/python-sdk

    Sends TTS audio to Anam and receives synchronized
    avatar video and audio back.
    """

    name = "anam_avatar"

    def __init__(
        self,
        avatar_id: str | None = None,
        api_key: str | None = None,
        client_options: ClientOptions | None = None,
        connect_timeout: float | None = None,
        session_ready_timeout: float | None = None,
        width: int = 1920,
        height: int = 1080,
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
        """
        api_key = api_key or os.getenv("ANAM_API_KEY")
        if not api_key:
            raise ValueError("Anam API key not provided")
        avatar_id = avatar_id or os.getenv("ANAM_AVATAR_ID")
        if not avatar_id:
            raise ValueError("Anam avatar ID not provided")

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
            width=width, height=height, fps=30, max_queue_size=30 * 20
        )

        self._connect_timeout = connect_timeout
        self._session_ready_timeout = session_ready_timeout

        self._connected = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._exit_stack = contextlib.AsyncExitStack()
        self._real_session: Session | None = None
        self._real_agent: Agent | None = None
        self._audio_input_stream: AgentAudioInputStream | None = None
        self._send_lock = asyncio.Lock()
        self._audio_receiver_task: asyncio.Task[None] | None = None
        self._video_receiver_task: asyncio.Task[None] | None = None

    def publish_video_track(self) -> QueuedVideoTrack:
        """Return the video track that receives avatar video frames."""
        return self._sync.video_track

    def publish_audio_track(self) -> AudioStreamTrack:
        """Return the audio track that receives avatar audio frames."""
        return self._sync.audio_track

    def attach_agent(self, agent: Agent) -> None:
        """
        Attach a running Agent to the Avatar publisher and subscribe
        to incoming audio events.

        Args:
            agent: Agent to attach.
        """
        self._real_agent = agent
        self._subscribe_to_audio_events()

    async def start(self) -> None:
        """Connect to Anam. Called by Agent via _apply("start") during join()."""
        await self._connect()

    async def close(self) -> None:
        """
        Close the Anam avatar publisher, cancel audio & video processing tasks
        and release resources.
        """
        for task in (self._audio_receiver_task, self._video_receiver_task):
            if task is not None:
                await cancel_and_wait(task)

        try:
            await self._exit_stack.aclose()
            await self._client.close()
        except Exception:
            logger.warning("Failed to close Anam avatar publisher", exc_info=True)
        finally:
            logger.debug("Anam avatar publisher closed")

    @property
    def _session(self) -> Session:
        if self._real_session is None:
            raise RuntimeError("Anam avatar session not initialized")
        return self._real_session

    @property
    def _agent(self) -> Agent:
        if self._real_agent is None:
            raise RuntimeError("Agent is not attached yet")
        return self._real_agent

    def _is_stale_epoch(self, epoch: int) -> bool:
        """Check if an event's epoch is behind the current LLM epoch."""
        llm = self._agent.llm
        return isinstance(llm, Realtime) and epoch != llm.epoch

    async def _video_receiver(self) -> None:
        async for frame in self._session.video_frames():
            try:
                await self._sync.write_video(frame)
            except Exception:
                logger.warning("Failed to write video frame", exc_info=True)

    async def _audio_receiver(self) -> None:
        async for frame in self._session.audio_frames():
            try:
                await self._sync.write_audio(PcmData.from_av_frame(frame))
            except Exception:
                logger.warning("Failed to send audio frame", exc_info=True)

    async def _send_audio(self, pcm: PcmData) -> None:
        if self._audio_input_stream is None:
            self._audio_input_stream = self._session.create_agent_audio_input_stream(
                AgentAudioInputConfig(
                    encoding="pcm_s16le", sample_rate=24000, channels=1
                )
            )
        for chunk in pcm.resample(target_channels=1, target_sample_rate=24000).chunks(
            2400  # 100ms
        ):
            await self._audio_input_stream.send_audio_chunk(chunk.to_bytes())

    async def _flush_audio(self) -> None:
        if self._audio_input_stream is not None:
            await self._audio_input_stream.end_sequence()

    def _subscribe_to_audio_events(self) -> None:
        @self._agent.events.subscribe
        async def on_tts_audio(event: TTSAudioEvent):
            # Use the lock because TTS events arrive asynchronously
            async with self._send_lock:
                if event.data is not None:
                    await self._send_audio(event.data)
                if event.is_final_chunk:
                    await self._flush_audio()

        @self._agent.events.subscribe
        async def on_realtime_audio(event: RealtimeAudioOutputEvent):
            async with self._send_lock:
                if event.data is not None and not self._is_stale_epoch(event.epoch):
                    await self._send_audio(event.data)

        @self._agent.events.subscribe
        async def on_realtime_audio_done(event: RealtimeAudioOutputDoneEvent):
            async with self._send_lock:
                await self._flush_audio()
                if event.interrupted:
                    await self._sync.flush()
                    await self._session.interrupt()

        @self._agent.events.subscribe
        async def on_turn_started(event: TurnStartedEvent):
            if (
                event.participant
                and event.participant.user_id != self._agent.agent_user.id
            ):
                async with self._send_lock:
                    await self._sync.flush()
                    await self._session.interrupt()

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
