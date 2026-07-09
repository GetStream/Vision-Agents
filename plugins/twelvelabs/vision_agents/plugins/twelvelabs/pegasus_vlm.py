import asyncio
import io
import logging
import math
import os
import time
import uuid
from collections import deque
from fractions import Fraction
from typing import AsyncIterator, Deque, Optional

import aiortc
import av
from twelvelabs import TwelveLabs
from twelvelabs.types.stream_analyze_response import (
    StreamAnalyzeResponse,
    StreamAnalyzeResponse_TextGeneration,
)
from twelvelabs.types.video_context import VideoContext_AssetId
from vision_agents.core import llm
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.video_forwarder import VideoForwarder

logger = logging.getLogger(__name__)

# Pegasus requires a minimum resolution of 360x360. Frames are scaled up to this
# floor on encode so low-resolution webcam tracks are still accepted by the API.
_MIN_DIMENSION = 360

# Pegasus rejects clips shorter than this many seconds.
_MIN_CLIP_SECONDS = 4

_ASSET_POLL_INTERVAL_S = 1.0
_ASSET_READY_TIMEOUT_S = 120.0


class PegasusVLM(llm.VideoLLM):
    """Video understanding VLM backed by TwelveLabs Pegasus.

    Unlike frame-by-frame VLMs, Pegasus analyzes a short video clip, so it can
    reason about motion and events over time. Recent frames from the watched
    track are buffered, encoded into an MP4 clip on demand, uploaded to the
    TwelveLabs Assets API, and analyzed with the supplied prompt. The streamed
    answer can be vocalized by the agent's TTS service.

    Args:
        api_key: TwelveLabs API key. Falls back to the ``TWELVELABS_API_KEY``
            environment variable.
        model_name: Pegasus model identifier (default: ``"pegasus1.5"``).
        fps: Frame sampling rate for the buffered clip (default: 1.0).
        clip_seconds: Length of the clip analyzed per request. Pegasus requires
            at least 4 seconds of video (default: 5).
        max_tokens: Maximum tokens in the response. Pegasus requires at least
            512 (default: 512).
    """

    provider_name = "twelvelabs"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "pegasus1.5",
        fps: float = 1.0,
        clip_seconds: int = 5,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()

        api_key = api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "api_key is required (pass it directly or set TWELVELABS_API_KEY)"
            )
        if clip_seconds < 4:
            raise ValueError("clip_seconds must be at least 4 for Pegasus")
        if max_tokens < 512:
            raise ValueError("max_tokens must be at least 512 for Pegasus")

        self.model = model_name
        self.fps = fps
        self.clip_seconds = clip_seconds
        self.max_tokens = max_tokens
        self.client = TwelveLabs(api_key=api_key)

        # Buffer enough frames to cover the requested clip window. Round up so a
        # non-integer fps (e.g. 29.97) never yields a clip shorter than the
        # requested duration, which Pegasus rejects.
        self._max_frames = max(1, math.ceil(fps * clip_seconds))
        self._frame_buffer: Deque[av.VideoFrame] = deque(maxlen=self._max_frames)
        self._video_forwarder: Optional[VideoForwarder] = None

    async def watch_video_track(
        self,
        track: aiortc.mediastreams.MediaStreamTrack,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """Subscribe to a video track and buffer its frames."""
        if self._video_forwarder is not None and shared_forwarder is None:
            logger.warning("Video forwarder already running, stopping previous one")
            await self.stop_watching_video_track()

        if shared_forwarder is not None:
            # Detach our handler from any previous forwarder before re-registering
            # so handlers don't accumulate when the forwarder is reassigned.
            if (
                self._video_forwarder is not None
                and self._video_forwarder is not shared_forwarder
            ):
                await self._video_forwarder.remove_frame_handler(
                    self._on_frame_received
                )

            self._video_forwarder = shared_forwarder
            logger.info("🎥 Pegasus subscribing to shared VideoForwarder")
            self._video_forwarder.add_frame_handler(
                self._on_frame_received,
                fps=self.fps,
                name="twelvelabs_pegasus",
            )
        else:
            self._video_forwarder = VideoForwarder(
                input_track=track,  # type: ignore[arg-type]
                max_buffer=self._max_frames,
                fps=self.fps,
                name="twelvelabs_pegasus_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._on_frame_received)

    async def simple_response(
        self,
        text: str,
        participant: Optional[Participant] = None,
    ) -> AsyncIterator[LLMResponseDelta | LLMResponseFinal]:
        """Analyze the buffered clip and stream Pegasus' answer.

        Args:
            text: Prompt describing what to analyze in the clip.
            participant: Optional participant the request relates to.
        """
        request_start_time = time.perf_counter()
        inference_id = str(uuid.uuid4())

        frames = list(self._frame_buffer)
        if not frames:
            logger.warning("No frames available, skipping Pegasus analysis")
            yield LLMResponseFinal(original=None, text="")
            return

        first_token_time: Optional[float] = None
        text_chunks: list[str] = []
        sequence_number = 0
        asset_id: Optional[str] = None

        try:
            clip = await asyncio.to_thread(self._encode_clip, frames)
            asset = await asyncio.to_thread(
                self.client.assets.create,
                method="direct",
                file=("clip.mp4", clip, "video/mp4"),
            )
            if asset.id is None:
                raise RuntimeError("TwelveLabs asset upload returned no id")
            asset_id = asset.id

            try:
                # Wait and analyze inside this try so a failed/timed-out ready
                # poll still cleans up the uploaded asset.
                if asset.status != "ready":
                    await self._wait_for_asset_ready(asset_id)

                stream = await asyncio.to_thread(
                    self.client.analyze_stream,
                    model_name=self.model,
                    video=VideoContext_AssetId(type="asset_id", asset_id=asset_id),
                    prompt=text,
                    max_tokens=self.max_tokens,
                )

                sentinel: StreamAnalyzeResponse | None = None
                while True:
                    event = await asyncio.to_thread(next, stream, sentinel)
                    if event is sentinel:
                        break
                    if not isinstance(event, StreamAnalyzeResponse_TextGeneration):
                        continue
                    if event.text is None:
                        continue

                    chunk = event.text
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
            finally:
                # Best-effort cleanup of the uploaded asset; a cleanup failure
                # must not mask the analysis result or its error.
                await asyncio.to_thread(self._delete_asset, asset_id)

            total_text = "".join(text_chunks)
            latency_ms = (time.perf_counter() - request_start_time) * 1000
            final_ttft_ms: Optional[float] = None
            if first_token_time is not None:
                final_ttft_ms = (first_token_time - request_start_time) * 1000

            self.metrics.on_vlm_inference(
                provider=self.provider_name,
                model=self.model,
                latency_ms=latency_ms,
                frames_processed=len(frames),
            )

            logger.debug("Pegasus response received (%d chars)", len(total_text))

            yield LLMResponseFinal(
                original=None,
                text=total_text,
                item_id=inference_id,
                latency_ms=latency_ms,
                time_to_first_token_ms=final_ttft_ms,
                model=self.model,
            )

        except Exception as exc:
            logger.exception("Error analyzing clip with Pegasus")
            self.on_llm_error(error=exc)
            yield LLMResponseFinal(original=None, text="")
            return

    async def stop_watching_video_track(self) -> None:
        """Stop video forwarding."""
        if self._video_forwarder is not None:
            # Detach our handler rather than stopping the forwarder outright; the
            # forwarder stops itself once no handlers remain, which leaves a
            # shared forwarder running for other subscribers.
            await self._video_forwarder.remove_frame_handler(self._on_frame_received)
            self._video_forwarder = None
            logger.info("Stopped video forwarding")

    async def _on_frame_received(self, frame: av.VideoFrame) -> None:
        """Buffer the latest sampled frame."""
        self._frame_buffer.append(frame)

    async def _wait_for_asset_ready(self, asset_id: str) -> None:
        """Poll TwelveLabs until the uploaded asset can be analyzed."""
        deadline = time.perf_counter() + _ASSET_READY_TIMEOUT_S
        while time.perf_counter() < deadline:
            asset = await asyncio.to_thread(self.client.assets.retrieve, asset_id)
            if asset.status == "ready":
                return
            if asset.status == "failed":
                raise RuntimeError(f"TwelveLabs asset processing failed: id={asset_id}")
            await asyncio.sleep(_ASSET_POLL_INTERVAL_S)
        raise TimeoutError(
            f"TwelveLabs asset not ready after {_ASSET_READY_TIMEOUT_S}s: id={asset_id}"
        )

    def _delete_asset(self, asset_id: str) -> None:
        """Best-effort delete of an uploaded TwelveLabs asset.

        Uploaded assets persist until explicitly deleted, so we remove the clip
        once analysis finishes. ``force=True`` deletes it even if referenced.
        Failures are logged but never raised so cleanup can't mask the result.
        """
        try:
            self.client.assets.delete(asset_id, force=True)
        except Exception:
            logger.exception("Failed to delete TwelveLabs asset %s", asset_id)

    def _encode_clip(self, frames: list[av.VideoFrame]) -> bytes:
        """Encode buffered frames into an in-memory MP4 clip.

        The encoded clip is guaranteed to be at least ``_MIN_CLIP_SECONDS`` long;
        if too few frames are buffered the last frame is repeated to pad it out,
        since Pegasus rejects clips shorter than the minimum duration.
        """
        first = frames[0]
        width = max(first.width, _MIN_DIMENSION)
        height = max(first.height, _MIN_DIMENSION)
        # libx264 requires even dimensions.
        width += width % 2
        height += height % 2

        # Use an exact fractional rate so non-integer fps (e.g. 29.97) is encoded
        # without truncation. The clip's real duration is frame_count / rate, so
        # ensure we emit enough frames to reach the minimum duration floor.
        rate = Fraction(self.fps).limit_denominator(1000) or Fraction(1)
        min_frames = math.ceil(rate * _MIN_CLIP_SECONDS)
        if len(frames) < min_frames:
            frames = frames + [frames[-1]] * (min_frames - len(frames))

        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        try:
            stream = container.add_stream("libx264", rate=rate)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            frame_time_base = Fraction(rate.denominator, rate.numerator)
            for index, frame in enumerate(frames):
                reformatted = frame.reformat(width=width, height=height, format="rgb24")
                reformatted.pts = index
                reformatted.time_base = frame_time_base
                for packet in stream.encode(reformatted):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
        return buffer.getvalue()

    async def close(self) -> None:
        """Stop watching the video track and release resources."""
        await self.stop_watching_video_track()
