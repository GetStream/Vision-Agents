import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner
from vision_agents.plugins import roboflow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

"""
A voice agent that looks through the caller's camera.

Voice runs in the Go acceleration backend. This process joins the same call as the video
worker: it streams the camera to Roboflow Serverless Video Streaming, publishes the frames
back with boxes drawn on, and the model reads the detections through `get_video_state`.

`rfdetr-nano` is a COCO detector, enough to prove the pipeline on a cup or a laptop. Point
it at a flower Workflow with `workflow_id=` and `workspace=` in place of `model_id=`.
"""


async def create_agent(**kwargs) -> Agent:
    return Agent(
        config="flower_spotter",
        processors=[roboflow.RoboflowStreamingProcessor(model_id="rfdetr-nano")],
    )


async def join_call(agent: Agent, call_id: str, **kwargs) -> None:
    async with agent.join(call_id):
        await agent.responses.create("greet the user in one short sentence")


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
