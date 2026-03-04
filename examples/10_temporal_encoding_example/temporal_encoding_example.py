"""
Temporal Frame Encoding — compressing video motion into single images.

Standard vision models receive ~1 frame per second, losing all motion
between snapshots.  Temporal encoding composites multiple frames into a
single image that preserves motion direction, speed, and trajectory.

Three encoding modes (pass as ``temporal_encoding=`` on any VLM plugin):

  • "rgb"     — maps time to color channels (R=early, G=mid, B=late).
                Static areas are gray; moving objects show color trails.
  • "heatmap" — overlays a warm heatmap of accumulated motion on the
                latest frame.
  • "grid"    — arranges evenly-sampled frames in a grid.

Usage:

    uv run python examples/10_temporal_encoding_example/temporal_encoding_example.py
"""

import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    vlm = gemini.VLM(
        model="gemini-3-flash-preview",
        fps=5,
        frame_buffer_seconds=2,
        temporal_encoding="rgb",
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Temporal Vision AI", id="agent"),
        instructions=(
            "You are a video analysis assistant. You receive temporally-encoded "
            "video frames that show motion as color. Describe what you observe: "
            "what objects are present, what is moving, in which direction, and how fast."
        ),
        processors=[],
        llm=vlm,
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.llm.simple_response("Describe what you see and any motion you detect.")
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
