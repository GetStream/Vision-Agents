import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import aiortc
import av
import torch
from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import Participant
from PIL import Image
from transformers import AutoModelForCausalLM
from vision_agents.core import llm
from vision_agents.core.agents.agent_types import AgentOptions, default_agent_options
from vision_agents.core.llm.events import (
    LLMResponseChunkEvent,
    LLMResponseCompletedEvent,
)
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.processors import Processor
from vision_agents.core.stt.events import STTTranscriptEvent
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
        max_workers: Number of worker threads for async operations (default: 10)
        force_cpu: If True, force CPU usage even if CUDA/MPS is available (default: False).
                  Auto-detects CUDA, then MPS (Apple Silicon), then defaults to CPU.
                  Note: MPS is automatically converted to CPU due to model compatibility. We recommend running on CUDA for best performance.
        model_name: Hugging Face model identifier (default: "moondream/moondream3-preview")
        options: AgentOptions for model directory configuration.
                If not provided, uses default_agent_options()
    """

    def __init__(
        self,
        mode: Literal["vqa", "caption"] = "vqa",
        max_workers: int = 10,
        force_cpu: bool = False,
        model_name: str = "moondream/moondream3-preview",
        options: Optional[AgentOptions] = None,
    ):
        super().__init__()

        self.max_workers = max_workers
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
        self._stt_subscription_setup = False
        self._processing_lock = asyncio.Lock()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.model = None

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
        self.model = model

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
        """Setup video forwarding and STT subscription."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self._stop_watching_video_track()

        if self.model is None:
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
        """Consume the generator stream from model query/caption methods."""
        chunks = []
        for chunk in generator:
            if isinstance(chunk, str):
                chunks.append(chunk)
            else:
                logger.warning(f"Unexpected chunk type: {type(chunk)}, value: {chunk}")
                if chunk:
                    chunks.append(str(chunk))
        result = "".join(chunks)
        return result

    async def _process_frame(
        self, text: Optional[str] = None
    ) -> Optional[LLMResponseEvent]:
        if self._latest_frame is None:
            logger.warning("No frames available, skipping Moondream processing")
            return None

        if self.model is None:
            logger.warning("Model not loaded, skipping Moondream processing")
            return None

        if self._processing_lock.locked():
            logger.debug("Moondream processing already in progress, skipping")
            return None

        try:
            await self._processing_lock.acquire()
        except Exception as e:
            logger.warning(f"Failed to acquire lock: {e}")
            return None

        latest_frame = self._latest_frame

        try:
            frame_array = latest_frame.to_ndarray(format="rgb24")
            image = Image.fromarray(frame_array)

            if self.mode == "vqa":
                if not text:
                    logger.warning("VQA mode requires text/question")
                    return None

                result = await asyncio.to_thread(
                    self.model.query, image, text, stream=True
                )

                if isinstance(result, dict) and "answer" in result:
                    stream = result["answer"]
                else:
                    stream = result

                answer = await asyncio.to_thread(self._consume_stream, stream)

                if not answer:
                    logger.warning("Moondream query returned empty answer")
                    return None

                self.events.send(LLMResponseChunkEvent(delta=answer))
                self.events.send(LLMResponseCompletedEvent(text=answer))
                logger.info(f"Moondream VQA response: {answer}")
                return LLMResponseEvent(original=answer, text=answer)

            else:
                result = await asyncio.to_thread(
                    self.model.caption, image, length="normal", stream=True
                )

                if isinstance(result, dict) and "caption" in result:
                    stream = result["caption"]
                else:
                    stream = result

                caption = await asyncio.to_thread(self._consume_stream, stream)

                if not caption:
                    logger.warning("Moondream caption returned empty result")
                    return None

                self.events.send(LLMResponseChunkEvent(delta=caption))
                self.events.send(LLMResponseCompletedEvent(text=caption))
                logger.info(f"Moondream caption: {caption}")
                return LLMResponseEvent(original=caption, text=caption)

        except Exception as e:
            logger.exception(f"Error processing frame: {e}")
            return LLMResponseEvent(original=None, text="", exception=e)
        finally:
            if self._processing_lock.locked():
                self._processing_lock.release()

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

    def close(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
        if self.model is not None:
            del self.model
            self.model = None
        logger.info("🛑 Moondream Local VLM closed")
