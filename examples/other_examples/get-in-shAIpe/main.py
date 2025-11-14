import asyncio
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
    # We'll create the callback after the agent is created so it can capture the agent reference
    agent = None  # Placeholder for closure
    
    # Define callback as a closure that will capture the agent
    # This allows the callback to access the agent instance
    async def on_squat_complete(rep_count: int, knee_angle: float, timestamp: float) -> None:
        """
        Callback function invoked when a squat is completed.
        This closure has access to the 'agent' variable from the outer scope.
        
        Args:
            rep_count: The total number of squats completed
            knee_angle: The knee angle at completion (in degrees)
            timestamp: The timestamp when the squat was completed
        """
        nonlocal agent  # Reference the agent from outer scope
        logger.info(f"🎉 Squat completion callback: Rep #{rep_count} completed at {timestamp:.3f} with knee angle {knee_angle:.1f}°")
        
        # Send both chat message and voice output when a squat is completed
        if agent is not None:
            try:
                # Create a motivational message
                if rep_count == 1:
                    message = f"Great job! You've completed your first squat! Keep it up!"
                elif rep_count % 5 == 0:
                    message = f"Excellent work! You've completed {rep_count} squats! You're doing amazing!"
                else:
                    message = f"Nice! Rep {rep_count} complete! Keep going!"
                
                # Send chat message to conversation
                if agent.conversation is not None:
                    agent_user_id = agent.agent_user.id or "agent"
                    await agent.conversation.send_message(
                        role="assistant",
                        user_id=agent_user_id,
                        content=message
                    )
                    logger.info(f"📨 Sent chat message: {message}")
                
                # Send voice output via LLM (works with Gemini Realtime and other realtime LLMs)
                if agent.llm is not None:
                    await agent.llm.simple_response(text=message)
                    logger.info(f"📢 Spoke message via voice: {message}")
            except Exception as e:
                logger.error(f"Error sending chat message or speaking: {e}", exc_info=True)
    
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
                on_squat_complete=on_squat_complete,
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
