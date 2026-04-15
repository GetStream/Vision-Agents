import asyncio
import logging
import os
from asyncio import CancelledError
from pathlib import Path
from typing import Optional, cast

import aiortc
import av
import websockets
from aiortc import MediaStreamTrack, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from decart import DecartClient, DecartSDKError, models
from decart.models import RealTimeModels
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.realtime.client import SetInput
from decart.types import ModelState, Prompt
from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder

from .decart_video_track import DecartVideoTrack

logger = logging.getLogger(__name__)

# Reference-image inputs accepted by RestylingProcessor. Matches the subset of
# decart.types.FileInput that both ModelState.image and SetInput.image support
# (bytes, str, Path) — HasRead is not accepted downstream, so we exclude it.
ImageInput = bytes | str | Path


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
                    model="lucy_2_rt"
                )
            ]
        )
    """

    name = "decart_restyling"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: RealTimeModels = "lucy_2_rt",
        initial_prompt: str = "Cyberpunk city",
        initial_image: Optional[ImageInput] = None,
        enhance: bool = True,
        mirror: bool = True,
        width: int = 1280,  # Model preferred
        height: int = 720,
        **kwargs,
    ):
        """Initialize the Decart restyling processor.

        Args:
            api_key: Decart API key. Uses DECART_API_KEY env var if not provided.
            model: Decart model name (default: "lucy_2_rt").
            initial_prompt: Initial style prompt text.
            initial_image: Optional reference image used on first connect. Accepts
                bytes, a file path, an http(s) URL, a data URI, or a raw base64 string.
            enhance: Whether to enhance the prompt (default: True).
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
        self.initial_image = initial_image
        self.enhance = enhance
        self.mirror = mirror
        self.width = width
        self.height = height

        self.model = models.realtime(self.model_name)

        self._decart_client = DecartClient(api_key=self.api_key, **kwargs)
        self._video_track = DecartVideoTrack(width=width, height=height)
        self._realtime_client: Optional[RealtimeClient] = None

        self._connected = False
        self._connecting = False
        self._connect_lock = asyncio.Lock()
        self._processing_task: Optional[asyncio.Task[None]] = None
        self._frame_receiving_task: Optional[asyncio.Task[None]] = None
        self._reconnect_task: Optional[asyncio.Task[None]] = None
        self._current_track: Optional[MediaStreamTrack] = None
        self._on_connection_change_callback = None

        logger.info(
            f"Decart RestylingProcessor initialized (model: {self.model_name}, prompt: {self.initial_prompt[:50]}...)"
        )

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        logger.info("Processing video track, connecting to Decart")
        self._current_track = incoming_track
        if not self._connected and not self._connecting:
            await self._connect_to_decart(incoming_track)

    def publish_video_track(self) -> VideoStreamTrack:
        return self._video_track

    async def update_state(
        self,
        prompt: Optional[str] = None,
        image: Optional[ImageInput] = None,
        enhance: Optional[bool] = None,
    ) -> None:
        """Atomically update the Decart state (prompt and/or reference image).

        Mirrors the JS SDK's ``realtimeClient.set({ prompt, enhance, image })``.
        At least one of ``prompt`` or ``image`` must be provided.

        Args:
            prompt: New style prompt. If None, the current prompt is unchanged.
            image: Reference image (bytes, file path, http(s) URL, data URI, or
                raw base64 string). If None, the current image is unchanged.
            enhance: Whether to enhance the prompt. Defaults to the instance's
                ``enhance`` attribute.
        """
        if prompt is None and image is None:
            raise ValueError("At least one of 'prompt' or 'image' must be provided")

        if not self._realtime_client:
            logger.debug("Cannot update state: not connected to Decart")
            return

        enhance_value = enhance if enhance is not None else self.enhance
        # SetInput.image accepts bytes | str only; convert Path -> str. The SDK
        # treats string inputs as a path if they exist on disk, otherwise as a
        # URL/data-URI/raw-base64.
        set_input_image: Optional[bytes | str] = (
            str(image) if isinstance(image, Path) else image
        )

        await self._realtime_client.set(
            SetInput(prompt=prompt, image=set_input_image, enhance=enhance_value)
        )

        if prompt is not None:
            self.initial_prompt = prompt
        if image is not None:
            self.initial_image = image

        logger.info(
            "Updated Decart state (prompt=%s, image=%s)",
            "<changed>" if prompt is not None else "<unchanged>",
            "<changed>" if image is not None else "<unchanged>",
        )

    async def update_prompt(
        self, prompt_text: str, enhance: Optional[bool] = None
    ) -> None:
        """Shortcut for ``update_state(prompt=prompt_text, enhance=enhance)``.

        Args:
            prompt_text: The text of the new prompt to be applied.
            enhance: Whether to enhance the prompt. Defaults to the instance's
                ``enhance`` attribute.
        """
        await self.update_state(prompt=prompt_text, enhance=enhance)

    async def set_mirror(self, enabled: bool) -> None:
        self.mirror = enabled
        logger.debug(f"Updated Decart mirror mode: {enabled}")

    async def _connect_to_decart(self, local_track: MediaStreamTrack) -> None:
        async with self._connect_lock:
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
                        enhance=self.enhance,
                    ),
                    image=self.initial_image,
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

                self._realtime_client.on(
                    "connection_change", self._on_connection_change
                )
                self._realtime_client.on("error", self._on_error)

                self._connected = True
                logger.info("Connected to Decart Realtime API")

                if self._processing_task is None or self._processing_task.done():
                    self._processing_task = asyncio.create_task(self._processing_loop())

            except (DecartSDKError, websockets.ConnectionClosedError, OSError) as e:
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
        except MediaStreamError:
            logger.debug("Decart media stream ended")
            self._connected = False
        except (DecartSDKError, websockets.ConnectionClosedError, OSError):
            logger.exception("Error receiving frames from Decart")
            self._connected = False

    def _on_connection_change(self, state: str) -> None:
        logger.info(f"Decart connection state changed: {state}")
        if state == "connected":
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
            self._reconnect_task = asyncio.create_task(
                self._connect_to_decart(self._current_track)
            )

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

    async def stop_processing(self) -> None:
        """Stop processing video when participant leaves."""
        if self._realtime_client:
            await self._disconnect_from_decart()
            logger.info("🛑 Stopped Decart video processing (participant left)")
        self._current_track = None

    async def close(self) -> None:
        await self.stop_processing()

        if self._video_track:
            self._video_track.stop()

        if self._frame_receiving_task and not self._frame_receiving_task.done():
            self._frame_receiving_task.cancel()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self._decart_client:
            await self._decart_client.close()
