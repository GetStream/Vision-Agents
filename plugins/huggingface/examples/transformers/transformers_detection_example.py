"""
Transformers Local Detection Example

Demonstrates running RT-DETRv2 object detection locally with Vision Agents.
The model runs directly on your hardware (Apache 2.0 licensed).

Creates an agent that uses:
- TransformersDetectionProcessor for local object detection (RT-DETRv2)
- OpenAI Realtime for LLM (handles audio I/O directly)
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable

First run will download the RT-DETRv2 model (~300MB).
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, huggingface, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with local Transformers object detection."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Detection Agent", id="agent"),
        instructions=(
            "You are an assistant that can see the user's video feed. "
            "Object detection is running on the video. Describe what "
            "objects are being detected. Respond concisely."
        ),
        processors=[
            huggingface.TransformersDetectionProcessor(
                model="PekingU/rtdetr_v2_r101vd",
                conf_threshold=0.5,
                fps=5,
            )
        ],
        llm=openai.Realtime(),
    )

    @agent.events.subscribe
    async def on_detection(event: huggingface.DetectionCompletedEvent):
        if event.objects:
            for obj in event.objects:
                logger.info(
                    f"Detected {obj['label']} at "
                    f"({obj['x1']}, {obj['y1']}, {obj['x2']}, {obj['y2']})"
                )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Transformers Detection Agent...")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
