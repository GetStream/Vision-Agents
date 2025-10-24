import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import aws, getstream, cartesia, deepgram, smart_turn

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Weather Bot"),
        instructions="You are a helpful weather bot. Use the provided tools to answer questions.",
        llm=aws.LLM(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            region_name="us-east-1"
        ),
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
        turn_detection=smart_turn.TurnDetection(buffer_duration=2.0, confidence_threshold=0.5),
    )

    # Register custom functions
    @agent.llm.register_function(
        name="get_weather",
        description="Get the current weather for a given city"
    )
    def get_weather(city: str) -> dict:
        """Get weather information for a city."""
        logger.info(f"Tool: get_weather called for city: {city}")
        if city.lower() == "boulder":
            return {"city": city, "temperature": 72, "condition": "Sunny"}
        return {"city": city, "temperature": "unknown", "condition": "unknown"}

    @agent.llm.register_function(
        name="calculate",
        description="Performs a mathematical calculation"
    )
    def calculate(expression: str) -> dict:
        """Performs a mathematical calculation."""
        logger.info(f"Tool: calculate called with expression: {expression}")
        try:
            result = eval(expression)  # DANGER: In a real app, use a safer math evaluator!
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}

    await agent.create_user()

    call = agent.edge.client.video.call("default", str(uuid4()))
    await agent.edge.open_demo(call)
    
    with await agent.join(call):
        # Give the agent a moment to connect
        await asyncio.sleep(5)
        
        # Test function calling with weather
        logger.info("Testing weather function...")
        await agent.llm.simple_response(
            text="What's the weather like in Boulder? Please use the get_weather function."
        )
        
        await asyncio.sleep(5)
        
        # Test function calling with calculation
        logger.info("Testing calculation function...")
        await agent.llm.simple_response(
            text="Can you calculate 25 multiplied by 4 using the calculate function?"
        )
        
        await asyncio.sleep(5)
        
        # Wait a bit before finishing
        await asyncio.sleep(5)
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())

