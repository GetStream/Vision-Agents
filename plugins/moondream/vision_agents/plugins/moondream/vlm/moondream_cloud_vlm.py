import asyncio
import logging
import os
from typing import Optional, List, Literal
from concurrent.futures import ThreadPoolExecutor

import aiortc
import av
from PIL import Image

from vision_agents.core import llm
from vision_agents.core.stt.events import STTTranscriptEvent
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_queue import VideoLatestNQueue
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
import moondream as md

logger = logging.getLogger(__name__)


class CloudVLM(llm.VideoLLM):
    """Cloud-hosted VLM using Moondream model for captioning or visual queries.

    This VLM sends frames to the hosted Moondream model to perform either captioning
    or visual question answering. The instructions are taken from the STT service and
    sent to the model along with the frame. Once the model has an output, the results
    are then vocalized with the supplied TTS service.

    Args:
        api_key: API key for Moondream Cloud API. If not provided, will attempt to read
                from MOONDREAM_API_KEY environment variable.
        mode: "vqa" for visual question answering or "caption" for image captioning (default: "vqa")
        max_workers: Number of worker threads for async operations (default: 10)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: Literal["vqa", "caption"] = "vqa",  # Default to VQA
        max_workers: int = 10,
    ):
        super().__init__()

        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        self.max_workers = max_workers
        self.mode = mode

        # Frame buffer using VideoLatestNQueue (maintains last 10 frames)
        self._frame_buffer: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(
            maxlen=10
        )
        # Keep latest frame reference for fast synchronous access
        self._latest_frame: Optional[av.VideoFrame] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._stt_subscription_setup = False
        self._processing_lock = asyncio.Lock()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self._load_model()

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Setup video forwarding and STT subscription."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self._stop_watching_video_track()

        if shared_forwarder is not None:
            # Use shared forwarder
            self._video_forwarder = shared_forwarder
            logger.info("🎥 Moondream subscribing to shared VideoForwarder")
            self._video_forwarder.add_frame_handler(
                self._on_frame_received,
                fps=1.0,  # Low FPS for VLM
                name="moondream_vlm",
            )
        else:
            # Create our own VideoForwarder
            self._video_forwarder = VideoForwarder(
                input_track=track,  # type: ignore[arg-type]
                max_buffer=10,
                fps=1.0,  # Low FPS for VLM
                name="moondream_vlm_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._on_frame_received)

        # Setup STT subscription (only once)
        if not self._stt_subscription_setup and self.agent:
            self._setup_stt_subscription()
            self._stt_subscription_setup = True

    async def _on_frame_received(self, frame: av.VideoFrame):
        """Callback to receive frames and add to buffer."""
        try:
            self._frame_buffer.put_latest_nowait(frame)
            self._latest_frame = frame
        except Exception as e:
            logger.error(f"Error adding frame to buffer: {e}")

    def _setup_stt_subscription(self):
        if not self.agent:
            logger.warning("Cannot setup STT subscription: agent not set")
            return

        @self.agent.events.subscribe
        async def on_stt_transcript(event: STTTranscriptEvent):
            await self._on_stt_transcript(event)

    def _consume_stream(self, generator):
        chunks = []
        for chunk in generator:
            logger.debug(f"Moondream stream chunk: {type(chunk)} - {chunk}")
            if isinstance(chunk, str):
                chunks.append(chunk)
            else:
                # Log unexpected types but continue processing
                logger.warning(f"Unexpected chunk type: {type(chunk)}, value: {chunk}")
                if chunk:
                    chunks.append(str(chunk))
        result = "".join(chunks)
        logger.debug(f"Moondream stream result: {result}")
        return result

    async def _process_frame(
        self, text: Optional[str] = None
    ) -> Optional[LLMResponseEvent]:
        if self._latest_frame is None:
            logger.warning("No frames available, skipping Moondream processing")
            return None

        if self._processing_lock.locked():
            logger.debug("Moondream processing already in progress, skipping")
            return None

        latest_frame = self._latest_frame

        async with self._processing_lock:
            try:
                # Convert frame to PIL Image
                frame_array = latest_frame.to_ndarray(format="rgb24")
                image = Image.fromarray(frame_array)

                # Process based on mode
                if self.mode == "vqa":
                    if not text:
                        logger.warning("VQA mode requires text/question")
                        return None
                    # Moondream SDK returns {"answer": <generator>}, extract the generator
                    result = self.model.query(image, text, stream=True)
                    stream = result["answer"]
                    answer = await asyncio.to_thread(self._consume_stream, stream)

                    if not answer:
                        logger.warning("Moondream query returned empty answer")
                        return None

                    self.events.send(LLMResponseChunkEvent(delta=answer))
                    self.events.send(LLMResponseCompletedEvent(text=answer))
                    logger.info(f"Moondream VQA response: {answer}")
                    return LLMResponseEvent(original=answer, text=answer)

                elif self.mode == "caption":
                    # Moondream SDK returns {"caption": <generator>}, extract the generator
                    result = self.model.caption(image, length="normal", stream=True)
                    stream = result["caption"]
                    caption = await asyncio.to_thread(self._consume_stream, stream)

                    if not caption:
                        logger.warning("Moondream caption returned empty result")
                        return None

                    self.events.send(LLMResponseChunkEvent(delta=caption))
                    self.events.send(LLMResponseCompletedEvent(text=caption))
                    logger.info(f"Moondream caption: {caption}")
                    return LLMResponseEvent(original=caption, text=caption)
                else:
                    logger.error(f"Unknown mode: {self.mode}")
                    return None

            except Exception as e:
                logger.exception(f"Error processing frame: {e}")
                return LLMResponseEvent(original=None, text="", exception=e)

    async def _on_stt_transcript(self, event: STTTranscriptEvent):
        """Handle STT transcript event - process with Moondream."""
        if not event.text:
            return

        await self._process_frame(text=event.text)

    async def simple_response(
        self,
        text: str,
        processors: Optional[List[Processor]] = None,
        participant: Optional[Participant] = None,
    ) -> LLMResponseEvent:
        """
        simple_response is a standardized way to create a response.

        Args:
            text: The text/question to respond to
            processors: list of processors (which contain state) about the video/voice AI
            participant: optionally the participant object

        Examples:
            await llm.simple_response("What do you see in this image?")
        """
        result = await self._process_frame(text=text if self.mode == "vqa" else None)
        if result is None:
            return LLMResponseEvent(
                original=None,
                text="",
                exception=ValueError("No frame available or processing failed"),
            )
        return result

    async def _stop_watching_video_track(self) -> None:
        """Stop video forwarding."""
        if self._video_forwarder is not None:
            await self._video_forwarder.stop()
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

    def _load_model(self):
        try:
            # Validate API key
            if not self.api_key:
                raise ValueError("api_key is required for Moondream Cloud API")

            # Initialize cloud model
            self.model = md.vl(api_key=self.api_key)
            logger.info("✅ Moondream SDK initialized")

        except Exception as e:
            logger.exception(f"❌ Failed to load Moondream model: {e}")
            raise

    def close(self):
        """Clean up resources."""
        self._shutdown = True
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
        logger.info("🛑 Moondream Processor closed")
