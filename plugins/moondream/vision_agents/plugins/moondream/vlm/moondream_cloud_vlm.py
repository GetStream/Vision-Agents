import asyncio
import logging
import os
import time
import uuid
from typing import AsyncIterator, Literal, Optional

import aiortc
import av
import moondream as md
from PIL import Image
from vision_agents.core import llm
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_queue import VideoLatestNQueue

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

    provider_name = "moondream_cloud"

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

        self._load_model()

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Setup video forwarding and STT subscription."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self.stop_watching_video_track()

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

    async def _on_frame_received(self, frame: av.VideoFrame):
        """Callback to receive frames and add to buffer."""
        try:
            self._frame_buffer.put_latest_nowait(frame)
            self._latest_frame = frame
        except Exception as e:
            logger.error(f"Error adding frame to buffer: {e}")

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """
        Stream a response using the latest buffered video frame.

        Args:
            text: The text/question to respond to (required for VQA mode).
            participant: optionally the participant object.

        Examples:
            async for item in llm.simple_response("What do you see in this image?"):
                ...
        """
        request_start_time = time.perf_counter()
        inference_id = str(uuid.uuid4())

        if self._latest_frame is None:
            logger.warning("No frames available, skipping Moondream processing")
            yield LLMResponseFinal(original=None, text="")
            return

        if self.mode == "vqa" and not text:
            logger.warning("VQA mode requires text/question")
            yield LLMResponseFinal(original=None, text="")
            return

        if self.mode not in ("vqa", "caption"):
            logger.error(f"Unknown mode: {self.mode}")
            yield LLMResponseFinal(original=None, text="")
            return

        latest_frame = self._latest_frame

        first_token_time: Optional[float] = None
        text_chunks: list[str] = []
        sequence_number = 0
        sentinel = object()

        try:
            # Convert frame to PIL Image
            frame_array = latest_frame.to_ndarray(format="rgb24")
            image = Image.fromarray(frame_array)

            if self.mode == "vqa":
                # Moondream SDK returns {"answer": <generator>}, extract the generator
                result = await asyncio.to_thread(
                    self.model.query, image, text, stream=True
                )
                stream = result["answer"]
            else:  # caption
                # Moondream SDK returns {"caption": <generator>}, extract the generator
                result = await asyncio.to_thread(
                    self.model.caption, image, length="normal", stream=True
                )
                stream = result["caption"]

            while True:
                chunk = await asyncio.to_thread(next, stream, sentinel)
                if chunk is sentinel:
                    break

                if not isinstance(chunk, str):
                    # Log unexpected types but continue processing
                    logger.warning(
                        f"Unexpected chunk type: {type(chunk)}, value: {chunk}"
                    )
                    if not chunk:
                        continue
                    chunk = str(chunk)

                is_first = first_token_time is None
                ttft_ms: Optional[float] = None
                if is_first:
                    first_token_time = time.perf_counter()
                    ttft_ms = (first_token_time - request_start_time) * 1000

                text_chunks.append(chunk)
                yield LLMResponseDelta(
                    content_index=None,
                    item_id=inference_id,
                    output_index=0,
                    sequence_number=sequence_number,
                    delta=chunk,
                    is_first_chunk=is_first,
                    time_to_first_token_ms=ttft_ms,
                )
                sequence_number += 1

            total_text = "".join(text_chunks)
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            final_ttft_ms: Optional[float] = None
            if first_token_time is not None:
                final_ttft_ms = (first_token_time - request_start_time) * 1000

            self.metrics.on_vlm_inference(
                provider=self.provider_name,
                model="moondream-cloud",
                latency_ms=latency_ms,
                frames_processed=1,
            )

            logger.info(f"Moondream {self.mode} response: {total_text}")

            yield LLMResponseFinal(
                original=result,
                text=total_text,
                item_id=inference_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=final_ttft_ms,
                model="moondream-cloud",
            )

        except Exception as exc:
            logger.exception("Error processing frame")
            self.metrics.on_vlm_error(
                provider=self.provider_name,
                error_type=type(exc).__name__,
            )
            yield LLMResponseFinal(original=None, text="")
            return

    async def stop_watching_video_track(self) -> None:
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
        logger.info("🛑 Moondream Processor closed")
