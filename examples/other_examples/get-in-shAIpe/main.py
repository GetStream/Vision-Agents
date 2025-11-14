import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini
from squat_yolo_processor import SquatYOLOProcessor
from config import SQUAT_CONFIG, YOLO_CONFIG, LLM_CONFIG

# Configure logging to show timestamps and detailed info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    agent = Agent(
        edge=getstream.Edge(),  # use stream for edge video transport
        agent_user=User(name="AI squat coach"),
        instructions="Read @squat-coach.md",  # read the squat coach markdown instructions
        llm=gemini.Realtime(fps=LLM_CONFIG["fps"]),  # Careful with FPS can get expensive
        processors=[
            SquatYOLOProcessor(
                model_path=YOLO_CONFIG["model_path"],
                min_squat_angle=SQUAT_CONFIG["min_squat_angle"],
                max_standing_angle=SQUAT_CONFIG["max_standing_angle"],
                conf_threshold=SQUAT_CONFIG["confidence_threshold"],
                enable_hand_tracking=SQUAT_CONFIG["enable_hand_tracking"],
                enable_wrist_highlights=SQUAT_CONFIG["enable_wrist_highlights"],
                imgsz=YOLO_CONFIG["imgsz"],
                device=YOLO_CONFIG["device"],
                max_workers=YOLO_CONFIG["max_workers"],
                fps=YOLO_CONFIG["fps"],
                interval=YOLO_CONFIG["interval"],
            )
        ],  # realtime pose detection with intelligent squat counting
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    # ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    # join the call and open a demo env
    with await agent.join(call):
        # all LLMs support a simple_response method and a more advanced native method (so you can always use the latest LLM features)
        await agent.llm.simple_response(
            text="Say hi! I'm your AI squat coach. The system is automatically counting your squats and analyzing your form. Get ready to do some squats and I'll provide feedback!"
        )
        # Gemini's native API is available here
        # agent.llm.send_realtime_input(text="Hello world")
        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
