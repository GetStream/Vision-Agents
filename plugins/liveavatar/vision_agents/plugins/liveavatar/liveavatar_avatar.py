import asyncio
import logging
import os

import av
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.inference import (
    AudioOutputChunk,
    AudioOutputFlush,
    AudioOutputStream,
)
from vision_agents.core.avatars import Avatar, AVSynchronizer
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.utils.video_track import QueuedVideoTrack

from .liveavatar_client import LiveAvatarClient, Session
from .liveavatar_rtc_manager import LiveAvatarRTCManager
from .liveavatar_websocket import LiveAvatarWebSocket

logger = logging.getLogger(__name__)


def _task_done_callback(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.exception(
            "LiveAvatar background task %s failed",
            task.get_name(),
            exc_info=task.exception(),
        )


class LiveAvatar(Avatar):
    """LiveAvatar plugin (LITE mode, custom-agent integration path).

    References:
    - https://docs.liveavatar.com
    - https://docs.liveavatar.com/docs/lite-mode/integration-paths

    Sends TTS audio via the LiveAvatar media-server WebSocket and
    receives synchronized lip-synced video and audio from the room
    LiveAvatar provisions for the session.
    """

    provider_name = "liveavatar"

    def __init__(
        self,
        avatar_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        is_sandbox: bool = True,
        max_session_duration: int | None = None,
        video_quality: str = "high",
        video_encoding: str = "H264",
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Initialize the LiveAvatar plugin.

        Args:
            avatar_id: LiveAvatar avatar UUID. Falls back to LIVEAVATAR_AVATAR_ID.
            api_key: LiveAvatar API key. Falls back to LIVEAVATAR_API_KEY.
            base_url: Override the LiveAvatar API base URL.
            is_sandbox: Sandbox sessions don't burn credits but are duration-capped.
            max_session_duration: Session length cap in seconds; None for the API default.
            video_quality: One of "low", "medium", "high", "very_high".
                Default - `"high"`.
            video_encoding: One of "H264", "VP8".
            width: Output video width in pixels.
            height: Output video height in pixels.
        """
        super().__init__()

        api_key = api_key or os.getenv("LIVEAVATAR_API_KEY")
        if not api_key:
            raise ValueError(
                "LiveAvatar API key required. Set LIVEAVATAR_API_KEY or pass api_key."
            )
        avatar_id = avatar_id or os.getenv("LIVEAVATAR_AVATAR_ID")
        if not avatar_id:
            raise ValueError(
                "Avatar ID required. Set LIVEAVATAR_AVATAR_ID or pass avatar_id."
            )

        self._client = LiveAvatarClient(api_key=api_key, base_url=base_url)
        self._rtc_manager = LiveAvatarRTCManager(
            on_video=self._on_video_frame,
            on_audio=self._on_audio_frame,
            on_disconnect=self._on_disconnect,
        )
        self._sync = AVSynchronizer(width=width, height=height)

        self._avatar_id = avatar_id
        self._is_sandbox = is_sandbox
        self._max_session_duration = max_session_duration
        self._video_quality = video_quality
        self._video_encoding = video_encoding

        self._session: Session | None = None
        self._websocket: LiveAvatarWebSocket | None = None
        self._audio_input_task: asyncio.Task[None] | None = None
        self._connected = False

        logger.debug("LiveAvatar initialized (%dx%d)", width, height)

    def video_output(self) -> QueuedVideoTrack:
        """Return the video track that receives avatar video frames."""
        return self._sync.video_output

    def audio_output(self) -> AudioOutputStream:
        """Return the audio stream that receives avatar audio frames."""
        return self._sync.audio_output

    async def start(self) -> None:
        """Connect to LiveAvatar. Called by the Agent during ``join()``."""
        await self._connect()

    async def close(self) -> None:
        """Stop the session and release all resources."""
        if self._audio_input_task is not None:
            await cancel_and_wait(self._audio_input_task)

        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception as exc:
                logger.warning("Failed to close LiveAvatar websocket: %s", exc)

        try:
            await self._rtc_manager.close()
        except Exception as exc:
            logger.warning("Failed to close LiveAvatar RTC manager: %s", exc)

        # Close sync AFTER rtc_manager so its receive tasks can't write to a
        # closed stream during teardown.
        self._sync.close()

        if self._session is not None:
            try:
                await self._client.stop_session(session_id=self._session.session_id)
            except Exception as exc:
                logger.warning("Failed to stop LiveAvatar session: %s", exc)

        try:
            await self._client.close()
        finally:
            self._connected = False
            logger.debug("LiveAvatar closed")

    async def _process_audio_input(self) -> None:
        async for item in self.input_audio_stream:
            if isinstance(item, AudioOutputChunk):
                if item.data is not None and self._websocket is not None:
                    await self._websocket.send_audio_frame(item.data)
                if item.final and self._websocket is not None:
                    await self._websocket.end_turn()
            elif isinstance(item, AudioOutputFlush):
                if self._websocket is not None:
                    await self._websocket.interrupt()
                await self._sync.flush()

    async def _connect(self) -> None:
        token = await self._client.create_session_token(
            self._avatar_id,
            is_sandbox=self._is_sandbox,
            max_session_duration=self._max_session_duration,
            video_quality=self._video_quality,
            video_encoding=self._video_encoding,
        )
        self._session = await self._client.start_session(token.session_token)

        try:
            await self._rtc_manager.connect(
                self._session.livekit_url, self._session.livekit_agent_token
            )
            self._websocket = LiveAvatarWebSocket(self._session.ws_url)
            await self._websocket.connect()
        except Exception:
            await self._rtc_manager.close()
            if self._websocket is not None:
                await self._websocket.close()
            raise

        self._connected = True
        logger.info(
            "LiveAvatar connection established session_id=%s", self._session.session_id
        )

        if self._audio_input_task is None:
            self._audio_input_task = asyncio.create_task(self._process_audio_input())
            self._audio_input_task.add_done_callback(_task_done_callback)

    async def _on_video_frame(self, frame: av.VideoFrame) -> None:
        await self._sync.write_video(frame)

    async def _on_audio_frame(self, pcm: PcmData) -> None:
        await self._sync.write_audio(pcm)

    async def _on_disconnect(self) -> None:
        logger.info("LiveAvatar disconnected")
        self._connected = False
