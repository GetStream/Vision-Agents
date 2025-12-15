import asyncio
import logging
import os
from asyncio import CancelledError
from typing import Optional, cast

import aiortc
import av
import websockets
from aiortc import MediaStreamTrack, VideoStreamTrack
from decart import DecartClient, DecartSDKError, models
from decart.models import RealTimeModels
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.types import ModelState, Prompt
from vision_agents.core.processors.base_processor import VideoProcessorPublisher

from .decart_video_track import DecartVideoTrack

logger = logging.getLogger(__name__)


def _should_reconnect(exc: Exception) -> bool:
    if isinstance(exc, websockets.ConnectionClosedError):
        return True

    if isinstance(exc, DecartSDKError):
        error_msg = str(exc).lower()
        if (
            "connection" in error_msg
            or "disconnect" in error_msg
            or "timeout" in error_msg
        ):
            return True

    return False


class RestylingProcessor(VideoProcessorPublisher):
    """Decart Realtime restyling processor for transforming user video tracks.

    This processor accepts the user's local video track, sends it to Decart's
    Realtime API via websocket, receives transformed frames, and publishes them
    as a new video track.

    Example:
        agent = Agent(
            edge=getstream.Edge(),
            agent_user=User(name="Styled AI"),
            instructions="Be helpful",
            llm=gemini.Realtime(),
            processors=[
                decart.RestylingProcessor(
                    initial_prompt="Studio Ghibli animation style",
                    model="mirage_v2"
                )
            ]
        )
    """

    name = "decart_restyling"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: RealTimeModels = "mirage_v2",
        initial_prompt: str = "Cyberpunk city",
        enrich: bool = True,
        mirror: bool = True,
        width: int = 1280,  # Model preferred
        height: int = 720,
        **kwargs,
    ):
        """Initialize the Decart restyling processor.

        Args:
            api_key: Decart API key. Uses DECART_API_KEY env var if not provided.
            model: Decart model name (default: "mirage_v2").
            initial_prompt: Initial style prompt text.
            enrich: Whether to enrich prompt (default: True).
            mirror: Mirror mode for front camera (default: True).
            width: Output video width (default: 1280).
            height: Output video height (default: 720).
            **kwargs: Additional arguments passed to parent class.
        """

        self.api_key = api_key or os.getenv("DECART_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Decart API key is required. Set DECART_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model_name = model
        self.initial_prompt = initial_prompt
        self.enrich = enrich
        self.mirror = mirror
        self.width = width
        self.height = height

        self.model = models.realtime(self.model_name)

        self._decart_client = DecartClient(api_key=self.api_key, **kwargs)
        self._video_track = DecartVideoTrack(width=width, height=height)
        self._realtime_client: Optional[RealtimeClient] = None

        self._connected = False
        self._connecting = False
        self._processing_task: Optional[asyncio.Task] = None
        self._frame_receiving_task: Optional[asyncio.Task] = None
        self._current_track: Optional[MediaStreamTrack] = None
        self._on_connection_change_callback = None

        logger.info(
            f"Decart RestylingProcessor initialized (model: {self.model_name}, prompt: {self.initial_prompt[:50]}...)"
        )

    async def process_video(self, incoming_track: aiortc.VideoStreamTrack, *_):
        logger.info("Processing video track, connecting to Decart")
        self._current_track = incoming_track
        if not self._connected and not self._connecting:
            await self._connect_to_decart(incoming_track)

    def publish_video_track(self) -> VideoStreamTrack:
        return self._video_track

    async def update_prompt(
        self, prompt_text: str, enrich: Optional[bool] = None
    ) -> None:
        """
        Updates the prompt used for the Decart real-time client. This method allows
        changing the current prompt and optionally specifies whether to enrich the
        prompt content. The operation is performed asynchronously and requires an
        active connection to the Decart client.

        If the `enrich` parameter is not provided, the method uses the default
        `self.enrich` value.

        Parameters:
            prompt_text: str
                The text of the new prompt to be applied.
            enrich: Optional[bool]
                Specifies whether to enrich the prompt content. If not provided,
                defaults to the object's `enrich` attribute.

        Returns:
            None
        """
        if not self._realtime_client:
            logger.debug("Cannot set prompt: not connected to Decart")
            return

        enrich_value = enrich if enrich is not None else self.enrich
        await self._realtime_client.set_prompt(prompt_text, enrich=enrich_value)
        self.initial_prompt = prompt_text
        logger.info(f"Updated Decart prompt: {prompt_text[:50]}...")

    async def set_mirror(self, enabled: bool) -> None:
        if not self._realtime_client:
            logger.debug("Cannot set mirror: not connected to Decart")
            return

        await self._realtime_client.set_mirror(enabled)
        self.mirror = enabled
        logger.debug(f"Updated Decart mirror mode: {enabled}")

    async def _connect_to_decart(self, local_track: MediaStreamTrack) -> None:
        if self._connecting:
            logger.debug("Already connecting to Decart, skipping")
            return

        logger.info(f"Connecting to Decart Realtime API (model: {self.model_name})")
        self._connecting = True

        try:
            if self._realtime_client:
                await self._disconnect_from_decart()

            initial_state = ModelState(
                prompt=Prompt(
                    text=self.initial_prompt,
                    enrich=self.enrich,
                ),
                mirror=self.mirror,
            )

            self._realtime_client = await RealtimeClient.connect(
                base_url=self._decart_client.base_url,
                api_key=self._decart_client.api_key,
                local_track=local_track,
                options=RealtimeConnectOptions(
                    model=self.model,
                    on_remote_stream=self._on_remote_stream,
                    initial_state=initial_state,
                ),
            )

            self._realtime_client.on("connection_change", self._on_connection_change)
            self._realtime_client.on("error", self._on_error)

            self._connected = True
            logger.info("Connected to Decart Realtime API")

            if self._processing_task is None or self._processing_task.done():
                self._processing_task = asyncio.create_task(self._processing_loop())

        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to Decart: {e}")
            raise
        finally:
            self._connecting = False

    def _on_remote_stream(self, transformed_stream: MediaStreamTrack) -> None:
        if self._frame_receiving_task and not self._frame_receiving_task.done():
            self._frame_receiving_task.cancel()

        self._frame_receiving_task = asyncio.create_task(
            self._receive_frames_from_decart(transformed_stream)
        )
        logger.debug("Started receiving frames from Decart transformed stream")

    async def _receive_frames_from_decart(
        self, transformed_stream: MediaStreamTrack
    ) -> None:
        try:
            while not self._video_track.is_stopped:
                frame = await transformed_stream.recv()
                await self._video_track.add_frame(cast(av.VideoFrame, frame))
        except asyncio.CancelledError:
            logger.debug("Frame receiving from Decart cancelled")

    def _on_connection_change(self, state: str) -> None:
        logger.info(f"Decart connection state changed: {state}")
        if state in ("connected", "connecting"):
            self._connected = True
        elif state in ("disconnected", "error"):
            self._connected = False
            if state == "disconnected":
                logger.info("Disconnected from Decart Realtime API")
            elif state == "error":
                logger.error("Decart connection error occurred")

        if self._on_connection_change_callback:
            self._on_connection_change_callback(state)

    def _on_error(self, error: DecartSDKError) -> None:
        logger.error(f"Decart error: {error}")
        if _should_reconnect(error) and self._current_track:
            logger.info("Attempting to reconnect to Decart...")
            asyncio.create_task(self._connect_to_decart(self._current_track))

    # Reconnect to Decart if the connection is dropped
    async def _processing_loop(self) -> None:
        try:
            while True:
                if not self._connected and not self._connecting and self._current_track:
                    logger.debug("Connection lost, attempting to reconnect...")
                    await self._connect_to_decart(self._current_track)

                await asyncio.sleep(1.0)
        except CancelledError:
            logger.debug("Decart processing loop cancelled")

    async def _disconnect_from_decart(self) -> None:
        if self._realtime_client:
            logger.debug("Disconnecting from Decart Realtime API")
            await self._realtime_client.disconnect()
            self._realtime_client = None
            self._connected = False

    async def close(self) -> None:
        if self._video_track:
            self._video_track.stop()

        if self._frame_receiving_task and not self._frame_receiving_task.done():
            self._frame_receiving_task.cancel()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        else:
            if self._realtime_client or self._decart_client:
                await self._disconnect_from_decart()
                await self._decart_client.close()
