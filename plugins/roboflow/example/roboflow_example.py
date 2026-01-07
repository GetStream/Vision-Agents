"""
Roboflow Object Detection Example

This example demonstrates Roboflow object detection with Vision Agents.

The agent uses:
- Roboflow for real-time object detection (local RF-DETR model)
- GetStream for edge/real-time communication
- OpenAI for LLM

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable
"""

from dotenv import load_dotenv

from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, openai, roboflow

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with Roboflow object detection."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Vision Agent", id="agent"),
        instructions="You're a helpful AI assistant that can see and describe what's happening in the video.",
        processors=[
            roboflow.RoboflowLocalDetectionProcessor(
                classes=["person"],  # Detect people by default
                conf_threshold=0.5,
                fps=5,
            )
        ],
        llm=openai.Realtime(),
    )

    @agent.events.subscribe
    async def on_detection(event: roboflow.DetectionCompletedEvent):
        """Print when objects are detected."""
        if event.objects:
            for obj in event.objects:
                print(f"Detected {obj['label']} at ({obj['x1']}, {obj['y1']})")

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
