"""
Roboflow Video Moderation Example

Demonstrates real-time content moderation using a custom Roboflow model
trained to detect offensive gestures. Detected regions are censored with
a heavy Gaussian blur so they are hidden from view.

The agent uses:
- Roboflow local inference for gesture detection (custom-trained model)
- Stream's edge network for video/audio transport
- Gemini Flash Lite for LLM (verbal warnings)
- Deepgram for STT and TTS

Requirements:
- GOOGLE_API_KEY environment variable
- DEEPGRAM_API_KEY environment variable
- ROBOFLOW_API_KEY environment variable (for model download)
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging
import os
import time
import warnings
from typing import Optional

import av
import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from inference import get_model

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.tts.events import TTSSynthesisCompleteEvent
from vision_agents.core.turn_detection.events import TurnStartedEvent
from vision_agents.core.warmup import Warmable
import vision_agents.plugins.deepgram as deepgram
import vision_agents.plugins.gemini as gemini
import vision_agents.plugins.getstream as getstream
import vision_agents.plugins.roboflow as roboflow
from vision_agents.plugins.roboflow.events import (
    DetectedObject,
    DetectionCompletedEvent,
)
from vision_agents.plugins.roboflow.roboflow_cloud_processor import (
    RoboflowCloudDetectionProcessor,
)

# Suppress warnings for inference model backends we don't use.
os.environ.setdefault("CORE_MODEL_SAM_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_SAM3_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_GAZE_ENABLED", "False")
os.environ.setdefault("CORE_MODEL_YOLO_WORLD_ENABLED", "False")

# Silence noisy warnings from inference's transitive dependencies
warnings.filterwarnings("ignore", message="Importing from timm.models.layers")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="inference")

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


class LocalModerationProcessor(RoboflowCloudDetectionProcessor, Warmable[object]):
    """Moderation processor that runs inference locally instead of via cloud API.

    Subclasses RoboflowCloudDetectionProcessor to reuse video forwarding,
    event wiring, and frame lifecycle. Overrides _run_inference to use the
    Roboflow `inference` package, which downloads model weights once and
    runs detection on this machine — eliminating cloud round-trip latency.

    Implements Warmable so the model is downloaded and loaded during agent
    startup rather than on the first video frame.

    Frames are forwarded immediately so the video stream stays smooth.
    Inference runs in the background; the blur holds at the last known
    detection position until the next result arrives.
    """

    name = "roboflow_local_moderation"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._api_key = str(
            kwargs.get("api_key") or os.getenv("ROBOFLOW_API_KEY") or ""
        )
        self._local_model: object | None = None
        self._latest_detections: Optional[sv.Detections] = None
        self._latest_classes: dict[int, str] = {}
        self._inference_task: Optional[asyncio.Task[None]] = None

    async def on_warmup(self) -> object:
        """Pre-load the Roboflow model during agent startup."""
        logger.info(
            "⏳ Warming up moderation model %s (downloading weights)...", self.model_id
        )
        model = await asyncio.to_thread(
            get_model, model_id=self.model_id, api_key=self._api_key
        )
        logger.info("✅ Moderation model %s warmed up and ready", self.model_id)
        return model

    def on_warmed_up(self, resource: object) -> None:
        logger.info("🔥 Moderation model attached to processor")
        self._local_model = resource

    async def close(self) -> None:
        if self._inference_task is not None and not self._inference_task.done():
            self._inference_task.cancel()
        await super().close()

    async def _process_frame(self, frame: av.VideoFrame) -> None:
        if self._closed or self._local_model is None:
            return

        detections = self._latest_detections
        has_detections = (
            detections is not None
            and detections.class_id is not None
            and detections.class_id.size
        )
        needs_inference = self._inference_task is None or self._inference_task.done()

        if has_detections:
            # Blur + forward on thread (only path that needs heavy CPU work)
            def _blur_frame() -> tuple[av.VideoFrame, np.ndarray]:
                image = frame.to_ndarray(format="rgb24")
                censored = censor_regions(image, detections)
                out = av.VideoFrame.from_ndarray(censored)
                out.pts = frame.pts
                out.time_base = frame.time_base
                return out, image

            output_frame, image = await asyncio.to_thread(_blur_frame)
            await self._video_track.add_frame(output_frame)
            if needs_inference:
                self._inference_task = asyncio.create_task(
                    self._run_background_inference(image)
                )
        elif needs_inference:
            # No blur needed — forward raw frame immediately, then convert
            # to ndarray on thread only for inference
            await self._video_track.add_frame(frame)
            image = await asyncio.to_thread(lambda: frame.to_ndarray(format="rgb24"))
            self._inference_task = asyncio.create_task(
                self._run_background_inference(image)
            )
        else:
            # No blur, inference already in-flight — just forward, zero thread work
            await self._video_track.add_frame(frame)

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

        img_height, img_width = image.shape[0:2]

        has_objects = detections.class_id is not None and detections.class_id.size
        detected_objects = (
            [
                DetectedObject(label=classes[class_id], x1=x1, y1=y1, x2=x2, y2=y2)
                for class_id, (x1, y1, x2, y2) in zip(
                    detections.class_id, detections.xyxy.astype(float)
                )
            ]
            if has_objects
            else []
        )

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

    async def _run_inference(
        self, image: np.ndarray
    ) -> tuple[sv.Detections, dict[int, str]]:
        """Run local inference instead of cloud API call."""
        model = self._local_model
        if model is None:
            raise RuntimeError("object not loaded – call _ensure_model_loaded first")
        loop = asyncio.get_running_loop()

        def detect() -> tuple[sv.Detections, dict[int, str]]:
            result = model.infer(image, confidence=self.conf_threshold)
            predictions = result[0] if isinstance(result, list) else result

            class_ids_to_labels: dict[int, str] = {}
            if not predictions.predictions:
                return sv.Detections.empty(), class_ids_to_labels

            x1_list, y1_list, x2_list, y2_list = [], [], [], []
            confidences: list[float] = []
            class_ids: list[int] = []

            for pred in predictions.predictions:
                class_id = pred.class_id
                class_name = pred.class_name
                class_ids.append(class_id)
                class_ids_to_labels[class_id] = class_name

                x1 = int(pred.x - pred.width / 2)
                y1 = int(pred.y - pred.height / 2)
                x2 = int(pred.x + pred.width / 2)
                y2 = int(pred.y + pred.height / 2)

                x1_list.append(x1)
                y1_list.append(y1)
                x2_list.append(x2)
                y2_list.append(y2)
                confidences.append(pred.confidence)

            if class_ids:
                detections_obj = sv.Detections(
                    xyxy=np.array(list(zip(x1_list, y1_list, x2_list, y2_list))),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids),
                )
            else:
                detections_obj = sv.Detections.empty()
            return detections_obj, class_ids_to_labels

        return await loop.run_in_executor(self._executor, detect)


INSTRUCTIONS = """\
You are a friendly video call moderator. You sound like a real person, not a robot.

When you receive a moderation alert, respond with one casual but firm sentence. \
Sound natural, like you're talking to a friend — not reading a policy document. \
Never say "formal warning" or "unacceptable behavior". \
For example: "Hey, let's keep it clean, alright?" or "Come on, not cool."

If someone keeps doing it, sound more annoyed and disappointed, not more corporate.
Otherwise, be warm and conversational.\
"""


async def create_agent(**kwargs: object) -> Agent:
    """Create a moderation agent with local Roboflow gesture detection."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Moderator", id="agent"),
        instructions=INSTRUCTIONS,
        processors=[
            LocalModerationProcessor(
                model_id=MODEL_ID,
                api_url="https://serverless.roboflow.com",
                conf_threshold=0.4,
                fps=15,
            )
        ],
        llm=gemini.LLM(model="gemini-3.1-flash-lite-preview"),
        stt=deepgram.STT(),
        tts=deepgram.TTS(),
    )

    gesture_active = False
    escalation_count = 0
    warning_lock = asyncio.Lock()
    participant_user_id: Optional[str] = None

    @agent.events.subscribe
    async def _track_participant(event: TurnStartedEvent) -> None:
        nonlocal participant_user_id
        if event.participant and event.participant.user_id != agent.agent_user.id:
            participant_user_id = event.participant.user_id

    async def _wait_for_tts_playback() -> None:
        """Wait for TTS synthesis to finish, then wait for audio to play out."""
        done = asyncio.Event()
        duration_ms: float = 0.0

        @agent.tts.events.subscribe
        async def _on_complete(event: TTSSynthesisCompleteEvent) -> None:
            nonlocal duration_ms
            duration_ms = event.audio_duration_ms or 0.0
            done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=10.0)
        except TimeoutError:
            pass
        finally:
            agent.tts.events.unsubscribe(_on_complete)
        # Wait for audio to finish playing + network transit buffer
        await asyncio.sleep(duration_ms / 1000.0 + 0.5)

    @agent.events.subscribe
    async def on_detection(event: roboflow.DetectionCompletedEvent) -> None:
        nonlocal gesture_active, escalation_count
        if not event.objects:
            if gesture_active:
                gesture_active = False
                logger.info("Offensive gesture no longer detected")
            return

        if warning_lock.locked():
            return

        labels = [obj["label"] for obj in event.objects]
        logger.warning(
            "Offensive gesture detected: %s (%.0fms)",
            ", ".join(labels),
            event.inference_time_ms,
        )

        async with warning_lock:
            gesture_active = True
            escalation_count += 1

            # Interrupt any in-progress speech and flush the audio
            # track so the warning plays immediately. tts.interrupt()
            # stops synthesis but audio already on the track keeps
            # playing unless we flush it.
            if agent.tts:
                await agent.tts.interrupt()
            if agent._audio_track is not None:
                await agent._audio_track.flush()

            if escalation_count == 1:
                await agent.simple_response(
                    "Someone just made an inappropriate gesture — "
                    "this is the 1st time. "
                    "Call them out casually in one sentence."
                )
            elif escalation_count == 2:
                await agent.simple_response(
                    "Someone made an inappropriate gesture for the 2nd time. "
                    "Warn them sternly in one sentence that if they do it "
                    "again they will be removed from the call."
                )
            else:
                await agent.simple_response(
                    f"This is the {escalation_count}{'rd' if escalation_count == 3 else 'th'} time. "
                    "Tell them in one sentence that they're being removed "
                    "from the call for repeated inappropriate behavior."
                )

            # Wait for the TTS audio to actually finish playing
            # before releasing the lock or kicking.
            await _wait_for_tts_playback()

            if escalation_count >= 3 and participant_user_id:
                logger.info(
                    "Kicking user %s after %d offences",
                    participant_user_id,
                    escalation_count,
                )
                await agent.call.kick_user(
                    user_id=participant_user_id,
                    block=True,
                )

    return agent


async def join_call(
    agent: Agent, call_type: str, call_id: str, **kwargs: object
) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call=call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
