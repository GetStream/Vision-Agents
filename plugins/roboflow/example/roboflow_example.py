import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, roboflow, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = openai.Realtime()

    agent = Agent(
        edge=getstream.Edge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=User(name="AI Sports Commentator", id="agent"),
        instructions="Read @instructions.md",
        processors=[
            roboflow.RoboflowLocalDetectionProcessor(
                classes=["person", "sports ball"],
                conf_threshold=0.5,
                fps=5,
            )
        ],
        llm=llm,
    )

    @agent.events.subscribe
    async def on_detection_completed(event: roboflow.DetectionCompletedEvent):
        # React on detected objects here
        ...

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    with await agent.join(call):
        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
