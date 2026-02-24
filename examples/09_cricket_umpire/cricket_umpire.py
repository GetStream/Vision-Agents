import os
import logging
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from getstream import Stream as StreamClient
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, ultralytics

logger = logging.getLogger(__name__)
load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Third Umpire AI"),
        instructions="Read @cricket_umpire.md",
        llm=gemini.Realtime(fps=10),
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.llm.simple_response(
            text="Introduce yourself as the AI Third Umpire for women's cricket. Say you are ready to review decisions. Wait and watch the video feed, then analyze any cricket scenarios you see."
        )
        # Keep the agent alive by sending periodic prompts
        import asyncio
        while True:
            await asyncio.sleep(20)
            await agent.llm.simple_response(
                text="Continue watching the cricket video feed. If you see any cricket scenario unfold, analyze it immediately and give your verdict. Otherwise stay silent and keep watching."
            )


if __name__ == "__main__":
    launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
    Runner(launcher=launcher).cli()