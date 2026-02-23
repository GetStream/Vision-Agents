import asyncio
import logging
from typing import Any

import av
from getstream.video.rtc import audio_track
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.llm.events import RealtimeAudioOutputEvent
from vision_agents.core.processors.base_processor import AudioPublisher, VideoPublisher
from vision_agents.core.tts.events import TTSAudioEvent
from vision_agents.core.utils.video_track import QueuedVideoTrack

from .lemonslice_client import LemonSliceClient
from .lemonslice_rtc_manager import LemonSliceRTCManager

logger = logging.getLogger(__name__)


class LemonSliceAvatarPublisher(AudioPublisher, VideoPublisher):
    """LemonSlice avatar video and audio publisher.

    Sends TTS audio to LemonSlice over LiveKit and receives synchronized
    avatar video and audio back.

    For standard LLMs: LemonSlice provides both video and audio.
    For Realtime LLMs: LemonSlice provides video only; LLM provides audio.
    """

    name = "lemonslice_avatar"

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
        self._video_track = QueuedVideoTrack(width=width, height=height)
        self._audio_track = audio_track.AudioStreamTrack(
            sample_rate=48000, channels=2, format="s16"
        )

        self._connected = False
        self._agent: Any = None
        self._send_lock = asyncio.Lock()

        logger.debug(f"LemonSlice AvatarPublisher initialized ({width}x{height})")

    def publish_video_track(self) -> QueuedVideoTrack:
        return self._video_track

    def publish_audio_track(self) -> audio_track.AudioStreamTrack:
        return self._audio_track

    def attach_agent(self, agent: Any) -> None:
        self._agent = agent
        self._subscribe_to_audio_events()

    async def start(self) -> None:
        """Connect to LemonSlice. Called by Agent via _apply("start") during join()."""
        await self._connect()

    async def close(self) -> None:
        self._video_track.stop()
        try:
            await self._rtc_manager.close()
        except Exception as exc:
            logger.warning(f"Failed to close LemonSlice RTC manager: {exc}")
        finally:
            await self._client.close()
            self._connected = False
            logger.debug("LemonSlice avatar publisher closed")

    def _subscribe_to_audio_events(self) -> None:
        @self._agent.events.subscribe
        async def on_tts_audio(event: TTSAudioEvent):
            # Use the lock because TTS events arrive asynchronously
            async with self._send_lock:
                if event.data is not None:
                    await self._rtc_manager.send_audio(event.data)
                if event.is_final_chunk:
                    await self._rtc_manager.flush()

        @self._agent.events.subscribe
        async def on_realtime_audio(event: RealtimeAudioOutputEvent):
            if event.data is not None:
                await self._rtc_manager.send_audio(event.data)

    async def _connect(self) -> None:
        credentials = self._rtc_manager.generate_credentials()
        await self._rtc_manager.connect(credentials)
        try:
            await self._client.create_session(
                credentials.livekit_url, credentials.livekit_token
            )
            self._connected = True
            logger.info("LemonSlice avatar connection established")
        except Exception as exc:
            logger.exception("Failed to create a LemonSlice session")
            await self._rtc_manager.close()

    async def _on_video_frame(self, frame: av.VideoFrame) -> None:
        await self._video_track.add_frame(frame)

    async def _on_audio_frame(self, pcm: PcmData) -> None:
        await self._audio_track.write(pcm)

    async def _on_disconnect(self) -> None:
        logger.info("LemonSlice disconnected")
        self._connected = False
