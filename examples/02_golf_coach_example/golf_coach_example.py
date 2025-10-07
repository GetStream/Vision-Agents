import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.plugins import getstream, ultralytics, openai
from vision_agents.core.agents import Agent
from vision_agents.core.cli import start_dispatcher
from getstream import AsyncStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


async def start_agent() -> None:
    call_id = str(uuid4())
    client = AsyncStream()
    agent_user = await client.create_user(name="AI golf coach")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,
        instructions="Read @golf_coach.md",
        llm=openai.Realtime(fps=10),
        #llm=gemini.Realtime(fps=1), # Careful with FPS can get expensive
        processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")],
    )

    call = client.video.call("default", call_id)


    with await agent.join(call):
        await agent.edge.open_demo(call)
        await agent.llm.simple_response(text="Say hi. After the user does their golf swing offer helpful feedback.")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
