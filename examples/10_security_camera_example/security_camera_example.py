import logging
from typing import Dict, Any

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, gemini, elevenlabs

from security_camera_processor import SecurityCameraProcessor

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
        fps=5,  # Process 5 frames per second
        time_window=1800,  # 30 minutes in seconds
        thumbnail_size=80,  # Size of face thumbnails
        detection_interval=2.0,  # Detect faces every 2 seconds
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Security AI", id="agent"),
        instructions="""You're a security camera AI assistant with face recognition capabilities.
You can detect and recognize unique individuals, tracking when they arrive and leave.
You help monitor who visits and can answer questions about security activity.
Keep responses short and professional. You have access to unique visitor counts and detailed visit information from the last 30 minutes.""",
        processors=[security_processor],  # Add the security camera processor
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Register function for getting visitor count
    @llm.register_function(description="Get the number of unique visitors detected in the last 30 minutes")
    async def get_visitor_count() -> Dict[str, Any]:
        count = security_processor.get_visitor_count()
        state = security_processor.state()
        return {
            "unique_visitors": count,
            "total_detections": state["total_detections"],
            "time_window": f"{state['time_window_minutes']} minutes",
            "last_detection": state["last_detection_time"],
        }
    
    # Register function for getting detailed visitor information
    @llm.register_function(description="Get detailed information about all visitors including when they were first and last seen")
    async def get_visitor_details() -> Dict[str, Any]:
        details = security_processor.get_visitor_details()
        return {
            "visitors": details,
            "total_unique_visitors": len(details),
        }

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    with await agent.join(call):
        # Greet the user
        await agent.simple_response(
            "Hello! I'm your security camera AI with face recognition. I can detect and recognize unique individuals, "
            "tracking when they arrive and how many times they've been seen. You can ask me about visitors in the last 30 minutes!"
        )

        # Run until the call ends
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))

