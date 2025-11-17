import asyncio
import logging
import os
from asyncio import CancelledError
from typing import Any, Optional

import aiortc
import websockets
from aiortc import MediaStreamTrack, VideoStreamTrack
from decart import DecartClient, models
from decart import DecartSDKError
from decart.realtime import RealtimeClient, RealtimeConnectOptions
from decart.types import ModelState, Prompt

from vision_agents.core.processors.base_processor import (
    AudioVideoProcessor,
    VideoProcessorMixin,
    VideoPublisherMixin,
)
from .decart_video_track import DecartVideoTrack

logger = logging.getLogger(__name__)


def _should_reconnect(exc: Exception) -> bool:
    """
    Determine if a websocket error should trigger a reconnect.
    
    Args:
        exc: The exception that occurred.
        
    Returns:
        True if reconnection should be attempted, False otherwise.
    """
    if isinstance(exc, websockets.ConnectionClosedError):
        return True

    # Check for Decart SDK errors that indicate connection issues
    if isinstance(exc, DecartSDKError):
        error_msg = str(exc).lower()
        if "connection" in error_msg or "disconnect" in error_msg or "timeout" in error_msg:
            return True

    return False


class RestylingProcessor(AudioVideoProcessor, VideoProcessorMixin, VideoPublisherMixin):
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
            model: str = "mirage_v2",
            initial_prompt: str = "Studio Ghibli animation style",
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
        super().__init__(
            interval=0,
            receive_audio=False,
            receive_video=True,
            **kwargs,
        )

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

        try:
            self.model = models.realtime(self.model_name)
        except Exception as e:
            logger.error(f"Failed to get Decart model {self.model_name}: {e}")
            raise

        self._decart_client = DecartClient(api_key=self.api_key)
        self._video_track = DecartVideoTrack(width=width, height=height)
        self._realtime_client: Optional[RealtimeClient] = None

        self._connected = False
        self._connecting = False
        self._processing_task: Optional[asyncio.Task] = None
        self._frame_receiving_task: Optional[asyncio.Task] = None
        self._current_track: Optional[MediaStreamTrack] = None
        self._on_connection_change_callback = None

        logger.info(
            f"Decart RestylingProcessor initialized "
            f"(model: {self.model_name}, prompt: {self.initial_prompt})"
        )

    async def process_video(
            self,
            incoming_track: aiortc.mediastreams.MediaStreamTrack,
            participant: Any,
            shared_forwarder=None,
    ):
        """Process incoming video track by connecting to Decart Realtime API.
        
        This method is called by the agent when a user video track is added.
        It connects to Decart with the user's track and starts receiving
        transformed frames.
        """
        logger.debug("Decart process_video starting")

        self._current_track = incoming_track
        if not self._connected and not self._connecting:
            await self._connect_to_decart(incoming_track)

        logger.debug("Decart video processing pipeline started")

    def publish_video_track(self) -> VideoStreamTrack:
        """Return the transformed video track for publishing.
        
        Returns:
            DecartVideoTrack that publishes transformed frames.
        """
        logger.info("Decart publish_video_track called")
        return self._video_track

    async def set_prompt(self, prompt_text: str, enrich: Optional[bool] = None) -> None:
        """Change the style prompt for restyling.
        
        Args:
            prompt_text: New style prompt text.
            enrich: Whether to enrich the prompt. Uses instance default if None.
        """
        if not self._realtime_client:
            logger.warning("Cannot set prompt: not connected to Decart")
            return

        enrich_value = enrich if enrich is not None else self.enrich

        try:
            await self._realtime_client.set_prompt(prompt_text, enrich=enrich_value)
            self.initial_prompt = prompt_text
            logger.info(f"Updated Decart prompt to: {prompt_text}")
        except Exception as e:
            logger.error(f"Failed to set Decart prompt: {e}")
            # Try to reconnect if connection is lost
            if _should_reconnect(e) and self._current_track:
                await self._connect_to_decart(self._current_track)

    async def set_mirror(self, enabled: bool) -> None:
        """Toggle mirror mode.
        
        Args:
            enabled: Whether to enable mirror mode.
        """
        if not self._realtime_client:
            logger.warning("Cannot set mirror: not connected to Decart")
            return

        try:
            await self._realtime_client.set_mirror(enabled)
            self.mirror = enabled
            logger.info(f"Updated Decart mirror mode to: {enabled}")
        except Exception as e:
            logger.error(f"Failed to set Decart mirror mode: {e}")
            # Try to reconnect if connection is lost
            if _should_reconnect(e) and self._current_track:
                await self._connect_to_decart(self._current_track)

    async def _connect_to_decart(self, local_track: MediaStreamTrack) -> None:
        """Establish connection to Decart Realtime API.
        
        Args:
            local_track: The user's local MediaStream to send to Decart.
        """
        if self._connecting:
            logger.debug("Already connecting to Decart, skipping")
            return

        self._connecting = True

        try:
            if self._realtime_client:
                await self._disconnect_from_decart()

            logger.info(f"Connecting to Decart Realtime API (model: {self.model_name})")
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

            # Set up event handlers
            self._realtime_client.on("connection_change", self._on_connection_change)
            self._realtime_client.on("error", self._on_error)

            self._connected = True
            logger.info("Connected to Decart Realtime API")

            if self._processing_task is None or self._processing_task.done():
                self._processing_task = asyncio.create_task(self._processing_loop())

        except Exception as e:
            logger.error(f"Failed to connect to Decart: {e}")
            self._connected = False
            raise
        finally:
            self._connecting = False

    def _on_remote_stream(self, transformed_stream: MediaStreamTrack) -> None:
        if self._frame_receiving_task and not self._frame_receiving_task.done():
            self._frame_receiving_task.cancel()

        self._frame_receiving_task = asyncio.create_task(
            self._receive_frames_from_decart(transformed_stream)
        )
        logger.info("Started receiving frames from Decart transformed stream")

    async def _receive_frames_from_decart(self, transformed_stream: MediaStreamTrack) -> None:
        try:
            while not self._video_track._stopped:
                try:
                    # Receive frame from Decart's transformed stream
                    frame = await transformed_stream.recv()

                    # Add to our video track for publishing
                    await self._video_track.add_frame(frame)

                except Exception as e:
                    if not self._video_track._stopped:
                        logger.warning(f"Error receiving frame from Decart: {e}")
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("Frame receiving from Decart cancelled")
        except Exception as e:
            logger.error(f"Fatal error receiving frames from Decart: {e}")

    def _on_connection_change(self, state: str) -> None:
        logger.info(f"Decart connection state changed: {state}")

        if state in ("connected", "connecting"):
            self._connected = True
        elif state in ("disconnected", "error"):
            self._connected = False

        if self._on_connection_change_callback:
            try:
                self._on_connection_change_callback(state)
            except Exception as e:
                logger.warning(f"Error in connection change callback: {e}")

    def _on_error(self, error: DecartSDKError) -> None:
        logger.error(f"Decart error: {error}")

        if _should_reconnect(error) and self._current_track:
            logger.info("Attempting to reconnect to Decart...")
            asyncio.create_task(self._connect_to_decart(self._current_track))

    async def _processing_loop(self) -> None:
        logger.debug("Starting Decart processing loop")
        try:
            while True:
                try:
                    # Check connection state periodically
                    if not self._connected and not self._connecting and self._current_track:
                        logger.info("Connection lost, attempting to reconnect...")
                        await self._connect_to_decart(self._current_track)

                    # Sleep to avoid busy waiting
                    await asyncio.sleep(1.0)

                except websockets.ConnectionClosedError as e:
                    if not _should_reconnect(e):
                        raise e
                    # Reconnect
                    if self._current_track:
                        await self._connect_to_decart(self._current_track)
                except Exception as e:
                    logger.exception("Error in Decart processing loop")
                    await asyncio.sleep(1.0)

        except CancelledError:
            logger.debug("Decart processing loop cancelled")

    async def _disconnect_from_decart(self) -> None:
        if self._realtime_client:
            try:
                await self._realtime_client.disconnect()

                logger.info("Disconnected from Decart Realtime API")
            except Exception as e:
                logger.warning(f"Error disconnecting from Decart: {e}")
            finally:
                self._realtime_client = None
                self._connected = False

    def close(self) -> None:
        logger.info("Closing Decart RestylingProcessor")

        if self._video_track:
            self._video_track.stop()

        if self._frame_receiving_task and not self._frame_receiving_task.done():
            self._frame_receiving_task.cancel()

        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if self._realtime_client or self._decart_client:
                    asyncio.create_task(self._async_close())
            else:
                if self._realtime_client or self._decart_client:
                    loop.run_until_complete(self._async_close())
        except RuntimeError:
            logger.warning("No event loop available for async cleanup")

        logger.info("Decart RestylingProcessor closed")

    async def _async_close(self) -> None:
        try:
            if self._realtime_client:
                await self._disconnect_from_decart()

            if self._decart_client:
                try:
                    await self._decart_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Decart client: {e}")
        except Exception as e:
            logger.warning(f"Error in async close: {e}")
