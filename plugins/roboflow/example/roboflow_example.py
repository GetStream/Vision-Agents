import logging
import random

from dotenv import load_dotenv
from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, openai, roboflow

from utils import Debouncer

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

    # A list of questions to pick from when pinging the model
    questions = [
        "Provide an update on the situation on the football field.",
        "What has just happened?",
        "What is happenning on the field right now?",
    ]

    # Call LLM once in 4s max
    debouncer = Debouncer(8)

    @agent.events.subscribe
    async def on_detection_completed(event: roboflow.DetectionCompletedEvent):
        """
        Trigger an action when Roboflow detected objects on the video.

        This function will be called for every detection,
        so we use previously created Debouncer object to avoid calling the LLM too often.
        """

        ball_detected = bool(
            [obj for obj in event.objects if obj["label"] == "sports ball"]
        )
        # Ping LLM for a commentary only when the ball is detected and the call is not debounced.
        if ball_detected and debouncer:
            # Pick a question randomly from the list
            await agent.simple_response(random.choice(questions))

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    with await agent.join(call):
        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
