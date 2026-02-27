import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import gemini, getstream, ultralytics

logger = logging.getLogger(__name__)
load_dotenv()

# Global agent reference so HTTP endpoint can trigger it
active_agent: Agent | None = None


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Third Umpire DRS"),
        instructions="Read @cricket_umpire.md",
        llm=gemini.Realtime(fps=2),
        processors=[
            ultralytics.YOLOPoseProcessor(
                model_path="yolo11n-pose.pt",
                imgsz=256
            )
        ],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    global active_agent
    active_agent = agent
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.llm.simple_response(
            text="Say: Third Umpire DRS ready. Awaiting referral."
        )
        await agent.finish()
    active_agent = None


# Extra FastAPI app for review trigger endpoint
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

review_app = FastAPI(lifespan=lifespan)

@review_app.post("/review/{review_type}")
async def trigger_review(review_type: str):
    global active_agent
    if active_agent is None:
        return {"error": "Agent not connected"}
    
    if review_type == "lbw":
        prompt = "The on-field umpire has referred an LBW decision. Analyze what you can see in the current video feed. Check: 1) Did the ball pitch in line? 2) Was the impact in line with the stumps? 3) Was the ball going on to hit the stumps? Give your verdict in the required format: DECISION / REVIEW TYPE / REASON / CONFIDENCE"
    else:
        prompt = "The on-field umpire has referred a Run Out decision. Analyze what you can see in the current video feed. Check: 1) Was the bat grounded before the stumps were broken? 2) Was any part of the body behind the crease? Give your verdict in the required format: DECISION / REVIEW TYPE / REASON / CONFIDENCE"
    
    await active_agent.llm.simple_response(text=prompt)
    return {"status": "review triggered"}


if __name__ == "__main__":
    import threading
    import uvicorn

    def run_api():
        uvicorn.run(review_app, host="0.0.0.0", port=8002, log_level="warning")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    logger.info("🌐 Review API running on http://localhost:8002")

    launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
    Runner(launcher=launcher).cli()