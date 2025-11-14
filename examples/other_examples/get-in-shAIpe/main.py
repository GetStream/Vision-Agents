import asyncio
import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini, elevenlabs
# Alternative TTS providers (uncomment if you prefer):
# from vision_agents.plugins import cartesia  # Fast, low-latency TTS (requires CARTESIA_API_KEY)
# from vision_agents.plugins import kokoro  # Fast TTS, no API key needed (requires espeak-ng)
from squat_yolo_processor import SquatYOLOProcessor
from squat_events import SquatCompletedEvent
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
        tts=elevenlabs.TTS(),  # Fast TTS for voice responses (requires ELEVENLABS_API_KEY env var)
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
    
    # Register the custom squat event
    agent.events.register(SquatCompletedEvent)
    
    # Subscribe to squat completion events
    # Following the pattern from https://visionagents.ai/guides/event-system
    @agent.events.subscribe
    async def handle_squat_completed(event: SquatCompletedEvent):
        """
        Handle squat completion events from the processor.
        This runs in the main event loop context, so async operations work correctly.
        """
        logger.info(f"🎉 Squat completed: Rep #{event.rep_count} at {event.timestamp:.3f} with knee angle {event.knee_angle:.1f}°")
        
        try:
            # Send chat message to conversation
            if agent.conversation is not None:
                agent_user_id = agent.agent_user.id or "agent"
                await agent.conversation.send_message(
                    role="assistant",
                    user_id=agent_user_id,
                    content=f"SQUAT COUNTER: {event.rep_count}"
                )
                logger.info(f"📨 Sent squat counter: {event.rep_count}")
            
            # Send voice output - use agent.say() for fast TTS, or agent.llm.simple_response() for LLM voice
            # Create a motivational message
            if event.rep_count == 1:
                message = f"Great job! You've completed your first squat! Keep it up!"
            elif event.rep_count % 5 == 0:
                message = f"Excellent work! You've completed {event.rep_count} squats! You're doing amazing!"
            else:
                message = f"Nice! Rep {event.rep_count} complete! Keep going!"
            
            # Option 1: Use agent.say() - FAST (uses TTS if available, falls back to LLM if not)
            # This is much faster than simple_response because it uses dedicated TTS
            await agent.say(message)
            logger.info(f"📢 Spoke message via voice: {message}")
            
            # Option 2: Use agent.llm.simple_response() - SLOWER (goes through LLM for audio generation)
            # Uncomment this if you want to use LLM voice instead of TTS
            # await agent.llm.simple_response(text=message)
        except Exception as e:
            logger.error(f"Error handling squat completed event: {e}", exc_info=True)
    
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
