"""
Transformers Local Segmentation Example

Demonstrates running object detection + segmentation locally with Vision Agents.
Detection (RT-DETRv2) produces bounding boxes, then segmentation (SAM2) produces
pixel-precise masks for each detected object.

Creates an agent that uses:
- TransformersDetectionProcessor for local object detection (RT-DETRv2)
- TransformersSegmentationProcessor for local segmentation (SAM2)
- OpenAI Realtime for LLM (handles audio I/O directly)
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- OPENAI_API_KEY environment variable

First run will download the RT-DETRv2 (~300MB) and SAM2 (~150MB) models.
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, huggingface, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    """Create an agent with local detection + segmentation."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Segmentation Agent", id="agent"),
        instructions=(
            "You are an assistant that can see the user's video feed. "
            "Object detection and segmentation are running on the video. "
            "Describe what objects are being detected and segmented. "
            "Respond concisely."
        ),
        processors=[
            # Segmentation must be first — the agent publishes the first
            # video publisher's output, and this is the one with annotations.
            huggingface.TransformersSegmentationProcessor(
                model="facebook/sam2.1-hiera-tiny",
                fps=5,
                annotate=True,
                mask_opacity=0.35,
            ),
            huggingface.TransformersDetectionProcessor(
                model="PekingU/rtdetr_v2_r101vd",
                conf_threshold=0.5,
                fps=5,
                annotate=False,
            ),
        ],
        llm=openai.Realtime(),
    )

    @agent.events.subscribe
    async def on_segmentation(event: huggingface.SegmentationCompletedEvent):
        if event.objects:
            for obj in event.objects:
                logger.info(
                    f"Segmented {obj['label']} "
                    f"(IoU={obj['confidence']:.2f}, area={obj['mask_area']}px) "
                    f"at ({obj['x1']}, {obj['y1']}, {obj['x2']}, {obj['y2']})"
                )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Transformers Segmentation Agent...")

    async with agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
