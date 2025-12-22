import logging
from typing import Dict, Any, Optional

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, gemini, elevenlabs

from security_camera_processor import (
    SecurityCameraProcessor,
    PersonDetectedEvent,
    PackageDetectedEvent,
)

logger = logging.getLogger(__name__)

load_dotenv()

"""
Security Camera Demo with Face Detection

This example demonstrates:
- Real-time face detection from camera feed
- 30-minute sliding window of detected faces
- Video overlay with visitor count and face thumbnails
- LLM integration to answer questions about security activity

The processor detects faces, stores thumbnails, and displays them in a grid
on the right side of the video with a visitor count. The AI agent can answer
questions about how many people have visited.
"""


async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-2.5-flash-lite")

    # Create security camera processor
    security_processor = SecurityCameraProcessor(
        fps=30,  # Process 5 frames per second
        time_window=1800,  # 30 minutes in seconds
        thumbnail_size=80,  # Size of face thumbnails
        detection_interval=2.0,  # Detect faces every 2 seconds
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Security AI", id="agent"),
        instructions="Read @instructions.md",
        processors=[security_processor],  # Add the security camera processor
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Merge processor events with agent events so subscriptions work
    agent.events.merge(security_processor.events)

    # Register function for getting visitor count
    @llm.register_function(
        description="Get the number of unique visitors detected in the last 30 minutes. Always pass include_details=false for a quick count."
    )
    async def get_visitor_count(include_details: Optional[bool] = False) -> Dict[str, Any]:
        count = security_processor.get_visitor_count()
        state = security_processor.state()
        return {
            "unique_visitors": count,
            "total_detections": state["total_face_detections"],
            "time_window": f"{state['time_window_minutes']} minutes",
            "last_detection": state["last_face_detection_time"],
        }

    # Register function for getting detailed visitor information
    @llm.register_function(
        description="Get detailed information about all visitors including when they were first and last seen. Always pass include_timestamps=true."
    )
    async def get_visitor_details(include_timestamps: Optional[bool] = True) -> Dict[str, Any]:
        details = security_processor.get_visitor_details()
        return {
            "visitors": details,
            "total_unique_visitors": len(details),
        }

    # Register function for getting package count
    @llm.register_function(
        description="Get the number of unique packages detected in the last 30 minutes. Always pass include_details=false for a quick count."
    )
    async def get_package_count(include_details: Optional[bool] = False) -> Dict[str, Any]:
        count = security_processor.get_package_count()
        state = security_processor.state()
        return {
            "unique_packages": count,
            "total_package_detections": state["total_package_detections"],
            "time_window": f"{state['time_window_minutes']} minutes",
            "last_detection": state["last_package_detection_time"],
        }

    # Register function for getting detailed package information
    @llm.register_function(
        description="Get detailed information about all packages including when they were first and last seen, and confidence scores. Always pass include_confidence=true."
    )
    async def get_package_details(include_confidence: Optional[bool] = True) -> Dict[str, Any]:
        details = security_processor.get_package_details()
        return {
            "packages": details,
            "total_unique_packages": len(details),
        }

    # Register function for getting activity log
    @llm.register_function(
        description="Get the recent activity log showing what happened (people arriving, packages detected, etc.). Use this to answer questions like 'what happened?' or 'did anyone come by?'. Pass limit to control how many entries to return."
    )
    async def get_activity_log(limit: Optional[int] = 20) -> Dict[str, Any]:
        log = security_processor.get_activity_log(limit=limit or 20)
        return {
            "activity_log": log,
            "total_entries": len(log),
        }

    # Register function for remembering a face
    @llm.register_function(
        description="Register the current person's face with a name so they can be recognized in the future. Use when user says things like 'remember me as [name]' or 'my name is [name]'. Pass the name to remember."
    )
    async def remember_my_face(name: str) -> Dict[str, Any]:
        result = security_processor.register_current_face_as(name)
        return result

    # Register function for listing known faces
    @llm.register_function(
        description="Get a list of all registered/known faces that can be recognized by name. Pass include_timestamps=true to see when each face was registered."
    )
    async def get_known_faces(include_timestamps: Optional[bool] = True) -> Dict[str, Any]:
        faces = security_processor.get_known_faces()
        return {
            "known_faces": faces,
            "total_known": len(faces),
        }

    # Subscribe to detection events via the agent's merged event system
    @agent.events.subscribe
    async def on_person_detected(event: PersonDetectedEvent):
        if event.is_new:
            agent.logger.info(f"🚨 NEW PERSON ALERT: {event.face_id} detected!")
        else:
            agent.logger.info(f"👤 Returning visitor: {event.face_id} (seen {event.detection_count}x)")

    @agent.events.subscribe
    async def on_package_detected(event: PackageDetectedEvent):
        if event.is_new:
            agent.logger.info(f"📦 NEW PACKAGE ALERT: {event.package_id} detected! (confidence: {event.confidence:.2f})")
        else:
            agent.logger.info(f"📦 Package still there: {event.package_id} (seen {event.detection_count}x)")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    with await agent.join(call):
        # Greet the user
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
