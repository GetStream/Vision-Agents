import asyncio
import logging
import os
import time
import uuid
from typing import AsyncIterator, Literal, Optional

import aiortc
import av
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
from vision_agents.core import llm
from vision_agents.core.agents.agent_types import AgentOptions, default_agent_options
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_queue import VideoLatestNQueue
from vision_agents.core.warmup import Warmable
from vision_agents.plugins.moondream.moondream_utils import handle_device

logger = logging.getLogger(__name__)


class LocalVLM(llm.VideoLLM, Warmable):
    """Local VLM using Moondream model for captioning or visual queries.

    This VLM downloads and runs the moondream3-preview model locally from Hugging Face,
    providing captioning and visual question answering capabilities without requiring an API key.

    Note: The moondream3-preview model is gated and requires authentication:
    - Request access at https://huggingface.co/moondream/moondream3-preview
    - Once approved, authenticate using one of:
      - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here
      - Run: huggingface-cli login

    Args:
        mode: "vqa" for visual question answering or "caption" for image captioning (default: "vqa")
        force_cpu: If True, force CPU usage even if CUDA/MPS is available (default: False).
                  Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU.
                  Note: MPS is automatically converted to CPU due to model compatibility. We recommend running on CUDA for best performance.
        model_name: Hugging Face model identifier (default: "moondream/moondream3-preview")
        options: AgentOptions for model directory configuration.
                If not provided, uses default_agent_options()
    """

    provider_name = "moondream_local"

    def __init__(
        self,
        mode: Literal["vqa", "caption"] = "vqa",
        force_cpu: bool = False,
        model_name: str = "moondream/moondream3-preview",
        options: Optional[AgentOptions] = None,
    ):
        super().__init__()

        self.mode = mode
        self.model_name = model_name
        self.force_cpu = force_cpu

        if options is None:
            self.options = default_agent_options()
        else:
            self.options = options

        if torch.backends.mps.is_available():
            self.force_cpu = True
            logger.warning(
                "⚠️ MPS detected but using CPU (moondream model has CUDA dependencies incompatible with MPS)"
            )

        if self.force_cpu:
            self.device, self._dtype = torch.device("cpu"), torch.float32
        else:
            self.device, self._dtype = handle_device()

        self._frame_buffer: VideoLatestNQueue[av.VideoFrame] = VideoLatestNQueue(
            maxlen=10
        )
        self._latest_frame: Optional[av.VideoFrame] = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._processing_lock = asyncio.Lock()

        self._md_client: AutoModelForCausalLM | None = None

        logger.info("🌙 Moondream Local VLM initialized")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📝 Mode: {self.mode}")

    async def on_warmup(self):
        """
        Load the Moondream model from Hugging Face.
        """
        logger.info(f"Loading Moondream model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        model = await asyncio.to_thread(  # type: ignore[func-returns-value]
            lambda: self._load_model_sync()
        )
        logger.info("✅ Moondream model loaded")
        return model

    def on_warmed_up(self, model) -> None:
        self._md_client = model

    def _load_model_sync(self):
        """Synchronous model loading function run in thread pool."""
        try:
            # Check for Hugging Face token (required for gated models)
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                logger.warning(
                    "⚠️ HF_TOKEN environment variable not set. "
                    "This model requires authentication. "
                    "Set HF_TOKEN or run 'huggingface-cli login'"
                )

            load_kwargs = {
                "trust_remote_code": True,
                "cache_dir": self.options.model_dir,
            }

            if hf_token:
                load_kwargs["token"] = hf_token
            else:
                load_kwargs["token"] = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map={"": self.device},
                dtype=self._dtype,
                **load_kwargs,
            )

            if self.force_cpu:
                model.to("cpu")  # type: ignore[arg-type]
            model.eval()
            logger.info(f"✅ Model loaded on {self.device} device")

            try:
                model.compile()
            except Exception as compile_error:
                # If compilation fails, log and continue without compilation
                logger.warning(
                    f"⚠️ Model compilation failed, continuing without compilation: {compile_error}"
                )

            return model
        except Exception as e:
            error_msg = str(e)
            if (
                "gated repo" in error_msg.lower()
                or "403" in error_msg
                or "authorized" in error_msg.lower()
            ):
                logger.exception(
                    "❌ Failed to load Moondream model: Model requires authentication.\n"
                    "This model is gated and requires access approval:\n"
                    f"1. Visit https://huggingface.co/{self.model_name} to request access\n"
                    "2. Once approved, authenticate using one of:\n"
                    "   - Set HF_TOKEN environment variable: export HF_TOKEN=your_token_here\n"
                    "   - Run: huggingface-cli login\n"
                    f"Original error: {e}"
                )
            else:
                logger.exception(f"❌ Failed to load Moondream model: {e}")
            raise

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Setup video forwarding."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self.stop_watching_video_track()

        if self._md_client is None:
            raise ValueError("The model is not loaded yet")

        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            logger.info("🎥 Moondream Local VLM subscribing to shared VideoForwarder")
            self._video_forwarder.add_frame_handler(
                self._on_frame_received, fps=1.0, name="moondream_local_vlm"
            )
        else:
            self._video_forwarder = VideoForwarder(
                input_track=track,  # type: ignore[arg-type]
                max_buffer=10,
                fps=1.0,
                name="moondream_local_vlm_forwarder",
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

        if self._md_client is None:
            logger.warning("Model not loaded, skipping Moondream processing")
            yield LLMResponseFinal(original=None, text="")
            return

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

        if self._processing_lock.locked():
            logger.debug("Moondream processing already in progress, skipping")
            yield LLMResponseFinal(original=None, text="")
            return

        async with self._processing_lock:
            latest_frame = self._latest_frame

            first_token_time: Optional[float] = None
            text_chunks: list[str] = []
            sequence_number = 0
            sentinel = object()

            try:
                frame_array = latest_frame.to_ndarray(format="rgb24")
                image = Image.fromarray(frame_array)

                if self.mode == "vqa":
                    result = await asyncio.to_thread(
                        self._md_client.query, image, text, stream=True
                    )
                    stream = (
                        result["answer"]
                        if isinstance(result, dict) and "answer" in result
                        else result
                    )
                else:
                    result = await asyncio.to_thread(
                        self._md_client.caption, image, length="normal", stream=True
                    )
                    stream = (
                        result["caption"]
                        if isinstance(result, dict) and "caption" in result
                        else result
                    )

                while True:
                    chunk = await asyncio.to_thread(next, stream, sentinel)
                    if chunk is sentinel:
                        break

                    if not isinstance(chunk, str):
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
                    model=self.model_name,
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
                    model=self.model_name,
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

    def close(self):
        """Clean up resources."""
        if self._md_client is not None:
            del self._md_client
            self._md_client = None
        logger.info("🛑 Moondream Local VLM closed")
