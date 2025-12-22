"""Tiled detection processor that batches multiple frames into a single API call."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import aiortc
import av
import moondream as md
import numpy as np
from PIL import Image

from vision_agents.core.processors.base_processor import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.plugins.moondream.detection.moondream_video_track import (
    MoondreamVideoTrack,
)
from vision_agents.plugins.moondream.moondream_utils import (
    annotate_detections,
    parse_detection_bbox,
)

logger = logging.getLogger(__name__)

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class TiledDetectionProcessor(VideoProcessorPublisher):
    """Performs object detection by tiling multiple frames into a single image.

    This processor collects N frames, tiles them into a grid (e.g., 2x2),
    sends one API request to Moondream, then maps detections back to each
    frame. This allows higher effective FPS while staying within API rate limits.

    Args:
        api_key: API key for Moondream Cloud API.
        tile_grid: Tuple of (rows, cols) for the tile grid. Default (2, 2) = 4 frames.
        conf_threshold: Confidence threshold for detections (default: 0.3)
        detect_objects: Object(s) to detect.
        input_fps: Frame rate to collect frames at (default: 10)
        output_fps: Frame rate to output processed frames at (default: 2)
        max_workers: Number of worker threads (default: 10)
    """

    name = "moondream_tiled"

    def __init__(
        self,
        api_key: Optional[str] = None,
        tile_grid: Tuple[int, int] = (2, 2),
        conf_threshold: float = 0.3,
        detect_objects: Union[str, List[str]] = "person",
        input_fps: float = 10.0,
        output_fps: float = 2.0,
        max_workers: int = 10,
    ):
        self.api_key = api_key or os.getenv("MOONDREAM_API_KEY")
        self.tile_rows, self.tile_cols = tile_grid
        self.num_tiles = self.tile_rows * self.tile_cols
        self.conf_threshold = conf_threshold
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.max_workers = max_workers
        self._shutdown = False

        # Frame buffer to collect frames for tiling
        self._frame_buffer: List[av.VideoFrame] = []
        self._buffer_lock = asyncio.Lock()

        # Last detection results to apply to new frames while waiting for API
        self._last_detections: List[Dict[str, Any]] = []
        
        # Semaphore to limit concurrent API calls (matches Moondream's 2 RPS limit)
        self._api_semaphore: Optional[asyncio.Semaphore] = None
        
        # Queue for outputting frames in order
        self._output_queue: asyncio.Queue[Tuple[int, List[Tuple[av.VideoFrame, Dict[str, Any]]]]] = asyncio.Queue()
        self._next_output_batch: int = 0
        self._batch_counter: int = 0
        self._pending_batches: Dict[int, List[Tuple[av.VideoFrame, Dict[str, Any]]]] = {}
        self._output_task: Optional[asyncio.Task] = None

        # Normalize detect_objects to list
        if isinstance(detect_objects, str):
            self.detect_objects = [detect_objects]
        elif isinstance(detect_objects, list):
            if not all(isinstance(obj, str) for obj in detect_objects):
                raise ValueError("detect_objects must be str or list of strings")
            self.detect_objects = detect_objects
        else:
            raise ValueError("detect_objects must be str or list of strings")

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="moondream_tiled"
        )

        self._video_track: MoondreamVideoTrack = MoondreamVideoTrack()
        self._video_forwarder: Optional[VideoForwarder] = None

        self._load_model()

        logger.info(
            f"🌙 Moondream Tiled Processor initialized with {self.tile_rows}x{self.tile_cols} grid"
        )
        logger.info(f"🎯 Detection configured for objects: {self.detect_objects}")

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ):
        """Process incoming video track with tiled detection."""
        logger.info("✅ Moondream Tiled process_video starting")

        if shared_forwarder is not None:
            self._video_forwarder = shared_forwarder
            logger.info(
                f"🎥 Moondream Tiled subscribing to shared VideoForwarder at {self.input_fps} FPS"
            )
            self._video_forwarder.add_frame_handler(
                self._on_frame_received,
                fps=float(self.input_fps),
                name="moondream_tiled",
            )
        else:
            self._video_forwarder = VideoForwarder(
                incoming_track,  # type: ignore[arg-type]
                max_buffer=30,
                fps=self.input_fps,
                name="moondream_tiled_forwarder",
            )
            self._video_forwarder.add_frame_handler(self._on_frame_received)

        logger.info("✅ Moondream Tiled video processing pipeline started")

    def publish_video_track(self):
        logger.info("📹 publish_video_track called")
        return self._video_track

    def _load_model(self):
        if not self.api_key:
            raise ValueError("api_key is required for Moondream Cloud API")
        self.model = md.vl(api_key=self.api_key)
        logger.info("✅ Moondream SDK initialized")

    async def _on_frame_received(self, frame: av.VideoFrame):
        """Collect frames and process when we have enough for a tile."""
        # Initialize semaphore on first frame (needs event loop)
        if self._api_semaphore is None:
            self._api_semaphore = asyncio.Semaphore(2)  # Allow 2 concurrent API calls
            self._output_task = asyncio.create_task(self._output_frames_in_order())
        
        async with self._buffer_lock:
            self._frame_buffer.append(frame)

            # When we have enough frames, process them as a batch
            if len(self._frame_buffer) >= self.num_tiles:
                frames_to_process = self._frame_buffer[:self.num_tiles]
                self._frame_buffer = self._frame_buffer[self.num_tiles:]
                batch_id = self._batch_counter
                self._batch_counter += 1
                # Process in parallel - don't await, let multiple batches run concurrently
                asyncio.create_task(self._process_tiled_frames_parallel(frames_to_process, batch_id))

    async def _process_tiled_frames_parallel(self, frames: List[av.VideoFrame], batch_id: int):
        """Process a batch with semaphore to limit concurrent API calls."""
        async with self._api_semaphore:
            try:
                # Convert frames to numpy arrays
                frame_arrays = [f.to_ndarray(format="rgb24") for f in frames]

                # Create tiled image
                tiled_image, tile_info = self._create_tiled_image(frame_arrays)

                # Run detection on tiled image
                all_detections = await self._run_tiled_inference(tiled_image)

                # Map detections back to individual frames
                per_frame_detections = self._map_detections_to_frames(
                    all_detections, tile_info
                )

                # Cache the detections for reference
                self._last_detections = all_detections

                # Prepare frame+detection pairs
                frame_results = [
                    (frame, {"detections": detections})
                    for frame, detections in zip(frames, per_frame_detections, strict=False)
                ]

            except Exception as e:
                logger.exception(f"❌ Tiled processing failed for batch {batch_id}: {e}")
                # On error, output original frames without detections
                frame_results = [(frame, {"detections": []}) for frame in frames]

            # Store results and trigger output if this batch is next in line
            async with self._buffer_lock:
                self._pending_batches[batch_id] = frame_results
                await self._try_output_pending_batches()

    async def _try_output_pending_batches(self):
        """Output any pending batches that are ready (in order)."""
        while self._next_output_batch in self._pending_batches:
            frame_results = self._pending_batches.pop(self._next_output_batch)
            await self._output_queue.put((self._next_output_batch, frame_results))
            self._next_output_batch += 1

    async def _output_frames_in_order(self):
        """Background task that outputs frames from the queue at the target FPS."""
        frame_interval = 1.0 / self.output_fps
        
        while not self._shutdown:
            try:
                batch_id, frame_results = await asyncio.wait_for(
                    self._output_queue.get(), timeout=1.0
                )
                
                for i, (frame, results) in enumerate(frame_results):
                    await self._output_annotated_frame(frame, results)
                    # Add delay between frames (but not after the last one)
                    if i < len(frame_results) - 1:
                        await asyncio.sleep(frame_interval)
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"❌ Output task error: {e}")

    def _create_tiled_image(
        self, frame_arrays: List[np.ndarray]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create a tiled image from multiple frames.

        Returns the tiled image and metadata for mapping detections back.
        """
        if not frame_arrays:
            raise ValueError("No frames to tile")

        # Get dimensions from first frame
        h, w = frame_arrays[0].shape[:2]

        # Create empty canvas for tiled image
        tiled_h = h * self.tile_rows
        tiled_w = w * self.tile_cols
        tiled = np.zeros((tiled_h, tiled_w, 3), dtype=np.uint8)

        # Place each frame in its tile position
        tile_positions = []
        for idx, frame_array in enumerate(frame_arrays):
            row = idx // self.tile_cols
            col = idx % self.tile_cols
            y_start = row * h
            x_start = col * w

            # Resize frame if needed to match expected dimensions
            if frame_array.shape[:2] != (h, w):
                frame_array = np.array(
                    Image.fromarray(frame_array).resize((w, h), Image.Resampling.LANCZOS)
                )

            tiled[y_start : y_start + h, x_start : x_start + w] = frame_array
            tile_positions.append(
                {
                    "idx": idx,
                    "x_start": x_start,
                    "y_start": y_start,
                    "width": w,
                    "height": h,
                }
            )

        tile_info = {
            "tile_positions": tile_positions,
            "tiled_width": tiled_w,
            "tiled_height": tiled_h,
            "frame_width": w,
            "frame_height": h,
        }

        return tiled, tile_info

    async def _run_tiled_inference(
        self, tiled_array: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Run detection on the tiled image."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self._run_detection_sync, tiled_array
        )

    def _run_detection_sync(self, frame_array: np.ndarray) -> List[Dict]:
        """Synchronous detection on tiled image."""
        import time
        
        image = Image.fromarray(frame_array)

        if self._shutdown:
            return []

        all_detections = []
        
        # Combine all object types into a single query to minimize API calls
        # Moondream's detect() takes a single object string, so we need to call per object
        # but we can log timing to understand the bottleneck
        for object_type in self.detect_objects:
            start = time.time()
            logger.info(f"🔍 Detecting '{object_type}' on tiled image...")
            try:
                result = self.model.detect(image, object_type)
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect '{object_type}': {e}")
                continue
            elapsed = time.time() - start
            logger.info(f"✅ Detection for '{object_type}' took {elapsed:.2f}s")

            for obj in result.get("objects", []):
                detection = parse_detection_bbox(obj, object_type, self.conf_threshold)
                if detection:
                    all_detections.append(detection)

        logger.info(f"🔍 Tiled detection found {len(all_detections)} objects total")
        return all_detections

    def _map_detections_to_frames(
        self, detections: List[Dict[str, Any]], tile_info: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """Map detections from tiled image back to individual frames.

        Detections have normalized coordinates (0-1) relative to the tiled image.
        We need to determine which tile each detection belongs to and convert
        coordinates to be relative to that tile.
        """
        tile_positions = tile_info["tile_positions"]
        tiled_w = tile_info["tiled_width"]
        tiled_h = tile_info["tiled_height"]
        frame_w = tile_info["frame_width"]
        frame_h = tile_info["frame_height"]

        # Initialize per-frame detection lists
        per_frame: List[List[Dict[str, Any]]] = [[] for _ in tile_positions]

        for detection in detections:
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue

            x_min, y_min, x_max, y_max = bbox

            # Convert normalized coords to pixel coords in tiled image
            px_x_min = x_min * tiled_w
            px_y_min = y_min * tiled_h
            px_x_max = x_max * tiled_w
            px_y_max = y_max * tiled_h

            # Find center of detection to determine which tile it belongs to
            center_x = (px_x_min + px_x_max) / 2
            center_y = (px_y_min + px_y_max) / 2

            # Determine tile index based on center position
            tile_col = int(center_x // frame_w)
            tile_row = int(center_y // frame_h)
            tile_idx = tile_row * self.tile_cols + tile_col

            if tile_idx < 0 or tile_idx >= len(tile_positions):
                continue

            tile = tile_positions[tile_idx]

            # Convert coordinates to be relative to this tile (normalized 0-1)
            new_x_min = (px_x_min - tile["x_start"]) / frame_w
            new_y_min = (px_y_min - tile["y_start"]) / frame_h
            new_x_max = (px_x_max - tile["x_start"]) / frame_w
            new_y_max = (px_y_max - tile["y_start"]) / frame_h

            # Clamp to valid range
            new_x_min = max(0.0, min(1.0, new_x_min))
            new_y_min = max(0.0, min(1.0, new_y_min))
            new_x_max = max(0.0, min(1.0, new_x_max))
            new_y_max = max(0.0, min(1.0, new_y_max))

            mapped_detection = {
                "label": detection.get("label", "object"),
                "bbox": [new_x_min, new_y_min, new_x_max, new_y_max],
                "confidence": detection.get("confidence", 1.0),
            }
            per_frame[tile_idx].append(mapped_detection)

        return per_frame

    async def _output_annotated_frame(
        self, frame: av.VideoFrame, results: Dict[str, Any]
    ):
        """Annotate a frame with detections and output it."""
        try:
            frame_array = frame.to_ndarray(format="rgb24")

            if results.get("detections"):
                frame_array = annotate_detections(frame_array, results)

            processed_frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            await self._video_track.add_frame(processed_frame)
        except Exception as e:
            logger.exception(f"❌ Frame annotation failed: {e}")
            await self._video_track.add_frame(frame)

    async def close(self):
        """Clean up resources."""
        self._shutdown = True
        if self._output_task:
            self._output_task.cancel()
            try:
                await self._output_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=False)
        logger.info("🛑 Moondream Tiled Processor closed")

