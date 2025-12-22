import asyncio
import logging
from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, moondream, openai
from vision_agents.core.events import CallSessionParticipantJoinedEvent 

logger = logging.getLogger(__name__)

load_dotenv()

async def create_agent(**kwargs) -> Agent:
    # Tiled processor: collects 4 frames (2x2 grid), sends one API request,
    # maps detections back to each frame.
    # 
    # With 2 RPS API limit and parallel processing:
    # - input_fps=8 → collect 4 frames in 500ms per batch
    # - 2 batches can run in parallel → effectively 4 RPS worth of frames
    # - output_fps=8 → smooth playback matching input rate
    tiled_processor = moondream.TiledDetectionProcessor(
        detect_objects=["person"],  # Single object = 1 API call per batch
        tile_grid=(2, 2),  # 2x2 = 4 frames per API call
        input_fps=8.0,     # Capture at 8 FPS (4 frames = 500ms to fill a batch)
        output_fps=8.0,    # Output at 8 FPS to match input
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="My happy AI friend", id="agent"
        ),
        llm=openai.Realtime(fps=1),
        processors=[tiled_processor]
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)

    with await agent.join(call):
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))

