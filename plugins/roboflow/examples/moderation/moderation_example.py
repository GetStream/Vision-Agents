"""
Roboflow Video Moderation Example

Demonstrates real-time content moderation using a custom Roboflow model
trained to detect offensive gestures. Detected regions are censored with
a heavy Gaussian blur so they are hidden from other call participants.

The agent uses:
- Roboflow cloud inference for gesture detection (custom-trained model)
- GetStream for real-time video communication
- OpenAI for LLM (verbal warnings)

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable
- ROBOFLOW_API_KEY and ROBOFLOW_API_URL environment variables
"""

import asyncio
import logging
import time
from typing import Optional

import av
import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, openai, roboflow
from vision_agents.plugins.roboflow.events import DetectedObject, DetectionCompletedEvent
from vision_agents.plugins.roboflow.roboflow_cloud_processor import (
    RoboflowCloudDetectionProcessor,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Replace with your Roboflow model trained on offensive gesture detection.
# This uses a NAS (Neural Architecture Search) model — the slug format differs
# from the standard "project/version" pattern.
MODEL_ID = "stream-playground/the-finger-dataset-b5ewr-3-nas-gpu-6aa2ba"


def censor_regions(image: np.ndarray, detections: sv.Detections) -> np.ndarray:
    """Apply a heavy Gaussian blur over each detected bounding box."""
    censored = image.copy()
    for xyxy in detections.xyxy:
        x1, y1, x2, y2 = xyxy.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        region = censored[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Heavy blur — kernel must be odd, scale with region size
        kw = max(region.shape[1] // 3, 1) | 1
        kh = max(region.shape[0] // 3, 1) | 1
        blurred = cv2.GaussianBlur(region, (kw, kh), sigmaX=30, sigmaY=30)
        censored[y1:y2, x1:x2] = blurred

        # Red border around censored area
        cv2.rectangle(censored, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return censored


class ModerationProcessor(RoboflowCloudDetectionProcessor):
    """Cloud detection processor that censors detections instead of annotating them.

    Frames are forwarded immediately so the video stream stays smooth.
    Inference runs in the background; the blur holds at the last known
    detection position until the next result arrives.
    """

    name = "roboflow_moderation"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        # serverless.roboflow.com isn't in the SDK's known URL list, so it
        # defaults to the v1 API which doesn't work for serverless. Force v0.
        self._client.select_api_v0()
        self._latest_detections: Optional[sv.Detections] = None
        self._latest_classes: dict[int, str] = {}
        self._inference_task: Optional[asyncio.Task[None]] = None

    async def close(self) -> None:
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
        await super().close()

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        if self._closed:
            return

        image = frame.to_ndarray(format="rgb24")

        # Apply blur at last known detection positions and forward immediately
        if (
            self._latest_detections is not None
            and self._latest_detections.class_id is not None
            and self._latest_detections.class_id.size
        ):
            censored_image = censor_regions(image, self._latest_detections)
            output_frame = av.VideoFrame.from_ndarray(censored_image)
            output_frame.pts = frame.pts
            output_frame.time_base = frame.time_base
            await self._video_track.add_frame(output_frame)
        else:
            await self._video_track.add_frame(frame)

        # Kick off inference in the background if none is in-flight
        if self._inference_task is None or self._inference_task.done():
            self._inference_task = asyncio.create_task(
                self._run_background_inference(image)
            )

    async def _run_background_inference(self, image: np.ndarray) -> None:
        start_time = time.perf_counter()
        try:
            detections, classes = await self._run_inference(image)
        except Exception:
            logger.exception("Background inference failed")
            return

        inference_time_ms = (time.perf_counter() - start_time) * 1000
        self._latest_detections = detections
        self._latest_classes = classes

        if detections.class_id is None or not detections.class_id.size:
            return

        img_height, img_width = image.shape[0:2]
        detected_objects = [
            DetectedObject(label=classes[class_id], x1=x1, y1=y1, x2=x2, y2=y2)
            for class_id, (x1, y1, x2, y2) in zip(
                detections.class_id, detections.xyxy.astype(float)
            )
        ]

        self.events.send(
            DetectionCompletedEvent(
                plugin_name=self.name,
                raw_detections=detections,
                objects=detected_objects,
                image_width=img_width,
                image_height=img_height,
                inference_time_ms=inference_time_ms,
                model_id=self.model_id,
            )
        )


INSTRUCTIONS = """\
You are a video call moderator. Your job is to maintain a respectful environment.

When you detect an offensive gesture (you will receive detection events), respond
with a calm but firm verbal warning. For example:
- "I noticed an inappropriate gesture. Please keep things respectful."
- "Let's keep this call professional, please."

If the gesture persists across multiple frames, escalate your warning tone.
Otherwise, be friendly and helpful to participants.\
"""


async def create_agent(**kwargs) -> Agent:
    """Create a moderation agent with Roboflow gesture detection."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Moderator", id="agent"),
        instructions=INSTRUCTIONS,
        processors=[
            ModerationProcessor(
                model_id=MODEL_ID,
                api_url="https://serverless.roboflow.com",
                conf_threshold=0.4,
                fps=5,
            )
        ],
        llm=openai.Realtime(),
    )

    gesture_active = False
    escalation_count = 0

    @agent.events.subscribe
    async def on_detection(event: roboflow.DetectionCompletedEvent) -> None:
        nonlocal gesture_active, escalation_count
        if not event.objects:
            if gesture_active:
                gesture_active = False
                logger.info("Offensive gesture no longer detected")
            return

        labels = [obj["label"] for obj in event.objects]
        logger.warning(
            "Offensive gesture detected: %s (%.0fms)",
            ", ".join(labels),
            event.inference_time_ms,
        )

        if not gesture_active:
            gesture_active = True
            escalation_count += 1
            if escalation_count == 1:
                await agent.simple_response(
                    "ALERT: An offensive gesture was just detected. "
                    "Please issue a calm but firm verbal warning."
                )
            else:
                await agent.simple_response(
                    f"ALERT: An offensive gesture has been detected again "
                    f"(occurrence #{escalation_count}). Escalate your warning tone."
                )

    return agent


async def join_call(
    agent: Agent, call_type: str, call_id: str, **kwargs: object
) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
