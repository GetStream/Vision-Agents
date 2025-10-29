import asyncio
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import getstream, ultralytics, gemini

load_dotenv()


async def start_agent() -> None:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(name="AI golf coach"),
        instructions="Read @golf_coach.md",  # read the golf coach markdown instructions
        llm=gemini.Realtime(fps=3),  # Careful with FPS can get expensive
        # llm=openai.Realtime(fps=10), use this to switch to openai
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")
        ],  # realtime pose detection with yolo
    )

    # create a call, some other video networks call this a room
    call = agent.edge.client.video.call("default", str(uuid4()))

    # join the call and open a demo env
    with await agent.join(call):
        await agent.edge.open_demo(call)
        # all LLMs support a simple_response method and a more advanced native method (so you can always use the latest LLM features)
        await agent.llm.simple_response(
            text="Say hi. After the user does their golf swing offer helpful feedback."
        )
        # Gemini's native API is available here
        # agent.llm.send_realtime_input(text="Hello world")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_agent())
