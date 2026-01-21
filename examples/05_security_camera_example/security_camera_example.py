"""
Security Camera Demo with Face Detection

This example demonstrates:
- Real-time face detection from camera feed
- 30-minute sliding window of detected faces
- Video overlay with visitor count and face thumbnails
- LLM integration to answer questions about security activity
- Package theft detection with wanted poster generation
"""

import asyncio
import logging
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv
from poster_generator import generate_and_post_poster
from security_camera_processor import (
    PackageDetectedEvent,
    PackageDisappearedEvent,
    PersonDetectedEvent,
    PersonDisappearedEvent,
    SecurityCameraProcessor,
)
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

load_dotenv()

logger = logging.getLogger(__name__)


async def handle_package_theft(
    agent: Agent,
    face_image: np.ndarray,
    suspect_name: str,
    processor: SecurityCameraProcessor,
) -> None:
    """Generate a wanted poster, display it in the call, and post it to X."""
    await agent.say(
        f"Alert! Package stolen by {suspect_name}! Generating wanted poster."
    )

    poster_bytes, tweet_url = await generate_and_post_poster(
        face_image,
        suspect_name,
        post_to_x_enabled=True,
        tweet_caption=f'🚨 WANTED: {suspect_name} caught "stealing" a package! AI-powered security #VisionAgents',
    )

    if poster_bytes:
        with open(f"wanted_poster_{suspect_name}.png", "wb") as f:
            f.write(poster_bytes)
        agent.logger.info("✅ Wanted poster saved")

        # Share the poster in the video call for 8 seconds
        processor.share_image(poster_bytes, duration=8.0)
        await agent.say("Here's the wanted poster for the package thief!")

        if tweet_url:
            agent.logger.info(f"🐦 Posted to X: {tweet_url}")
            await agent.say("Wanted poster also posted to X!")
        else:
            agent.logger.warning("⚠️ Failed to post to X (check credentials)")
    else:
        agent.logger.warning(f"⚠️ Failed to generate wanted poster for {suspect_name}")


PACKAGE_THEFT_DELAY_SECONDS = 3.0

# Track pending theft checks - cancelled if package reappears
_pending_theft_tasks: Dict[str, asyncio.Task] = {}

# Track package history (since processor deletes packages when they disappear)
_package_history: Dict[
    str, Dict[str, Any]
] = {}  # package_id -> {first_seen, last_seen, detection_count, picked_up_by}


async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-2.5-flash-lite")

    # Create security camera processor
    security_processor = SecurityCameraProcessor(
        fps=5,
        time_window=1800,
        thumbnail_size=80,
        detection_interval=2.0,
        bbox_update_interval=0.3,  # Fast bbox updates for responsive face tracking
        model_path="weights_custom.pt",
        package_conf_threshold=0.7,
        max_tracked_packages=1,  # Single-package mode: always update existing package
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Security AI", id="agent"),
        instructions="Read @instructions.md",
        processors=[security_processor],
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Merge processor events with agent events so subscriptions work
    agent.events.merge(security_processor.events)

    @llm.register_function(
        description="Get the number of unique visitors detected in the last 30 minutes."
    )
    async def get_visitor_count() -> Dict[str, Any]:
        count = security_processor.get_visitor_count()
        state = security_processor.state()
        return {
            "unique_visitors": count,
            "total_detections": state["total_face_detections"],
            "time_window": f"{state['time_window_minutes']} minutes",
            "last_detection": state["last_face_detection_time"],
        }

    @llm.register_function(
        description="Get detailed information about all visitors including when they were first and last seen."
    )
    async def get_visitor_details() -> Dict[str, Any]:
        details = security_processor.get_visitor_details()
        return {
            "visitors": details,
            "total_unique_visitors": len(details),
        }

    @llm.register_function(
        description="Get package statistics including total packages seen and how many were picked up."
    )
    async def get_package_count() -> Dict[str, Any]:
        currently_visible = security_processor.get_package_count()
        total_seen = len(_package_history)
        picked_up = sum(1 for p in _package_history.values() if p.get("picked_up_by"))
        return {
            "currently_visible_packages": currently_visible,
            "total_packages_seen": total_seen,
            "packages_picked_up": picked_up,
        }

    @llm.register_function(
        description="Get detailed history of all packages seen, including who picked them up."
    )
    async def get_package_details() -> Dict[str, Any]:
        return {
            "packages": list(_package_history.values()),
            "total_packages_seen": len(_package_history),
        }

    @llm.register_function(
        description="Get recent activity log (people arriving, packages detected). Answers 'what happened?' or 'did anyone come by?'"
    )
    async def get_activity_log(limit: int = 20) -> Dict[str, Any]:
        log = security_processor.get_activity_log(limit=limit)
        return {"activity_log": log, "total_entries": len(log)}

    # Register function for remembering a face
    @llm.register_function(
        description="Register the current person's face with a name so they can be recognized in the future. Use when user says things like 'remember me as [name]' or 'my name is [name]'. Pass the name to remember."
    )
    async def remember_my_face(name: str) -> Dict[str, Any]:
        result = security_processor.register_current_face_as(name)
        return result

    @llm.register_function(
        description="Get a list of all registered faces that can be recognized by name."
    )
    async def get_known_faces() -> Dict[str, Any]:
        faces = security_processor.get_known_faces()
        return {"known_faces": faces, "total_known": len(faces)}

    # Subscribe to detection events via the agent's merged event system
    @agent.events.subscribe
    async def on_person_detected(event: PersonDetectedEvent):
        if event.is_new:
            agent.logger.info(f"🚨 NEW PERSON ALERT: {event.face_id} detected!")
        else:
            agent.logger.info(
                f"👤 Returning visitor: {event.face_id} (seen {event.detection_count}x)"
            )
            # Greet returning visitors
            await agent.say(f"Welcome back, {event.face_id}!")

    @agent.events.subscribe
    async def on_person_disappeared(event: PersonDisappearedEvent):
        display_name = event.name or event.face_id[:8]
        agent.logger.info(f"👤 Person left: {display_name}")

    @agent.events.subscribe
    async def on_package_detected(event: PackageDetectedEvent):
        # Cancel ALL pending theft checks when any package is detected.
        # This handles the case where the same physical package gets a new ID
        # after a brief confidence drop.
        if _pending_theft_tasks:
            cancelled_ids = list(_pending_theft_tasks.keys())
            for pkg_id in cancelled_ids:
                _pending_theft_tasks[pkg_id].cancel()
                del _pending_theft_tasks[pkg_id]
            agent.logger.info(
                f"📦 Package detected - cancelled theft checks for: {', '.join(cancelled_ids)}"
            )

        # Track package in our history
        if event.package_id not in _package_history:
            _package_history[event.package_id] = {
                "package_id": event.package_id,
                "first_seen": event.timestamp.isoformat(),
                "last_seen": event.timestamp.isoformat(),
                "detection_count": 1,
                "confidence": event.confidence,
                "picked_up_by": None,
            }
        else:
            _package_history[event.package_id]["last_seen"] = (
                event.timestamp.isoformat()
            )
            _package_history[event.package_id]["detection_count"] += 1

        if event.is_new:
            agent.logger.info(
                f"📦 NEW PACKAGE ALERT: {event.package_id} detected! (confidence: {event.confidence:.2f})"
            )
        else:
            agent.logger.info(
                f"📦 Package returned: {event.package_id} (visit #{event.detection_count})"
            )

    @agent.events.subscribe
    async def on_package_disappeared(event: PackageDisappearedEvent):
        picker_display = event.picker_name or (
            event.picker_face_id[:8] if event.picker_face_id else "unknown"
        )
        agent.logger.info(
            f"📦 Package {event.package_id} disappeared (suspect: {picker_display}) - "
            f"waiting {PACKAGE_THEFT_DELAY_SECONDS}s to confirm"
        )

        async def delayed_theft_check():
            await asyncio.sleep(PACKAGE_THEFT_DELAY_SECONDS)
            # If we get here, package didn't reappear
            del _pending_theft_tasks[event.package_id]
            agent.logger.info(
                f"📦 Package {event.package_id} confirmed gone - triggering theft workflow"
            )

            # Record who picked up the package in our history
            if event.package_id in _package_history:
                _package_history[event.package_id]["picked_up_by"] = picker_display

            if event.picker_face_id:
                face_image = security_processor.get_face_image(event.picker_face_id)
                if face_image is not None:
                    await handle_package_theft(
                        agent, face_image, picker_display, security_processor
                    )

        _pending_theft_tasks[event.package_id] = asyncio.create_task(
            delayed_theft_check()
        )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    async with agent.join(call):
        # Greet the user
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
