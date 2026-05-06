import asyncio
import logging
import os
from typing import Any

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

from .lemonslice_client import LemonSliceClient
from .lemonslice_rtc_manager import LemonSliceRTCManager

logger = logging.getLogger(__name__)


def _task_done_callback(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error(
            "LemonSlice avatar background task %s failed",
            task.get_name(),
            exc_info=task.exception(),
        )


class LemonSliceAvatar(Avatar):
    """LemonSlice avatar video and audio publisher.

    Sends TTS audio to LemonSlice over LiveKit and receives synchronized
    avatar video and audio back.

    For standard LLMs: LemonSlice provides both video and audio.
    For Realtime LLMs: LemonSlice provides video only; LLM provides audio.
    """

    provider_name = "lemonslice_avatar"

    def __init__(
        self,
        agent_id: str | None = None,
        agent_image_url: str | None = None,
        agent_prompt: str | None = None,
        idle_timeout: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        livekit_url: str | None = None,
        livekit_api_key: str | None = None,
        livekit_api_secret: str | None = None,
        width: int = 1920,
        height: int = 1080,
    ):
        """Initialize the LemonSlice avatar publisher.

        Args:
            agent_id: LemonSlice agent ID.
            agent_image_url: URL of the agent's avatar image.
            agent_prompt: Prompt describing the agent's persona.
            idle_timeout: Seconds before an idle session is closed.
            api_key: LemonSlice API key. Uses LEMONSLICE_API_KEY env var if not provided.
            base_url: LemonSlice API base URL override.
            livekit_url: LiveKit server URL. Uses LIVEKIT_URL env var if not provided.
            livekit_api_key: LiveKit API key. Uses LIVEKIT_API_KEY env var if not provided.
            livekit_api_secret: LiveKit API secret. Uses LIVEKIT_API_SECRET env var if not provided.
            width: Output video width in pixels.
            height: Output video height in pixels.
        """
        super().__init__()
        agent_id = agent_id or os.getenv("LEMONSLICE_AGENT_ID")
        agent_image_url = agent_image_url or os.getenv("LEMONSLICE_AGENT_IMAGE_URL")

        client_kwargs: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_image_url": agent_image_url,
            "agent_prompt": agent_prompt,
            "idle_timeout": idle_timeout,
            "api_key": api_key,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url

        self._client = LemonSliceClient(**client_kwargs)
        self._rtc_manager = LemonSliceRTCManager(
            on_video=self._on_video_frame,
            on_audio=self._on_audio_frame,
            on_disconnect=self._on_disconnect,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
        )
        self._sync = AVSynchronizer(width=width, height=height)

        self._connected = False
        self._audio_input_task: asyncio.Task[None] | None = None

        logger.debug(f"LemonSlice AvatarPublisher initialized ({width}x{height})")

    def video_output(self) -> QueuedVideoTrack:
        return self._sync.video_output

    def audio_output(self) -> AudioOutputStream:
        return self._sync.audio_output

    async def start(self) -> None:
        """Connect to LemonSlice. Called by Agent via _apply("start") during join()."""
        await self._connect()

    async def close(self) -> None:
        self._sync.close()
        if self._audio_input_task is not None:
            await cancel_and_wait(self._audio_input_task)

        try:
            await self._rtc_manager.close()
        except Exception as exc:
            logger.warning(f"Failed to close LemonSlice RTC manager: {exc}")
        finally:
            await self._client.close()
            self._connected = False
            logger.debug("LemonSlice avatar publisher closed")

    async def _process_audio_input(self) -> None:
        """
        Process audio input from the Agent
        """

        async for item in self.input_audio_stream:
            if isinstance(item, AudioOutputChunk):
                # Received normal audio, send it to the avatar
                if item.data is not None:
                    await self._rtc_manager.send_audio(item.data)
                # Received final audio chunk (end-of-utterance), flush avatar's audio
                if item.final:
                    await self._end_turn()

            elif isinstance(item, AudioOutputFlush):
                # Audio was interrupted
                await self._end_turn()
                await self._sync.flush()
                await self._rtc_manager.interrupt()

    async def _end_turn(self) -> None:
        await self._rtc_manager.flush()

    async def _connect(self) -> None:
        credentials = self._rtc_manager.generate_credentials()
        await self._rtc_manager.connect(credentials)
        try:
            await self._client.create_session(
                credentials.livekit_url, credentials.livekit_token
            )
        except Exception:
            await self._rtc_manager.close()
            raise
        self._connected = True
        logger.info("LemonSlice avatar connection established")

        if self._audio_input_task is None:
            self._audio_input_task = asyncio.create_task(self._process_audio_input())
            self._audio_input_task.add_done_callback(_task_done_callback)

    async def _on_video_frame(self, frame: av.VideoFrame) -> None:
        await self._sync.write_video(frame)

    async def _on_audio_frame(self, pcm: PcmData) -> None:
        await self._sync.write_audio(pcm)

    async def _on_disconnect(self) -> None:
        logger.info("LemonSlice disconnected")
        self._connected = False
