import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, ultralytics

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(name="AI golf coach"),
        instructions="Read @golf_coach.md",  # read the golf coach markdown instructions
        llm=gemini.Realtime(fps=3),  # Share video with gemini
        # llm=openai.Realtime(fps=3), # use this to switch to openai
        processors=[
            ultralytics.YOLOPoseProcessor(model_path="yolo26n-pose.pt")
        ],  # realtime pose detection with yolo
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # join the call and open a demo env
    async with agent.join(call):
        # all LLMs support a simple_response method and a more advanced native method (so you can always use the latest LLM features)
        await agent.llm.simple_response(
            text="Say hi. After the user does their golf swing offer helpful feedback."
        )
        # Gemini's native API is available here
        # agent.llm.send_realtime_input(text="Hello world")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
