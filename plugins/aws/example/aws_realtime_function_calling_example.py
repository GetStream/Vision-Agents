import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core import User
from vision_agents.core.agents import Agent
from vision_agents.plugins import aws, getstream

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [call_id=%(call_id)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def start_agent() -> None:
    """Example demonstrating AWS Bedrock realtime with function calling.
    
    This example creates an agent that can call custom functions to get
    weather information and perform calculations.
    """
    
    # Create the agent with AWS Bedrock Realtime
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Weather Assistant AI"),
        instructions="""You are a helpful weather assistant. When users ask about weather,
        use the get_weather function to fetch current conditions. You can also help with
        simple calculations using the calculate function.""",
        llm=aws.Realtime(
            model="amazon.nova-sonic-v1:0",
            region_name="us-east-1",
            voice_id="matthew"
        ),
    )
    
    # Register custom functions that the LLM can call
    @agent.llm.register_function(
        name="get_weather",
        description="Get the current weather for a given city"
    )
    def get_weather(city: str) -> dict:
        """Get weather information for a city.
        
        Args:
            city: The name of the city
            
        Returns:
            Weather information including temperature and conditions
        """
        # This is a mock implementation - in production you'd call a real weather API
        weather_data = {
            "Boulder": {"temp": 72, "condition": "Sunny", "humidity": 30},
            "Seattle": {"temp": 58, "condition": "Rainy", "humidity": 85},
            "Miami": {"temp": 85, "condition": "Partly Cloudy", "humidity": 70},
        }
        
        city_weather = weather_data.get(city, {"temp": 70, "condition": "Unknown", "humidity": 50})
        return {
            "city": city,
            "temperature": city_weather["temp"],
            "condition": city_weather["condition"],
            "humidity": city_weather["humidity"],
            "unit": "Fahrenheit"
        }
    
    @agent.llm.register_function(
        name="calculate",
        description="Perform a mathematical calculation"
    )
    def calculate(operation: str, a: float, b: float) -> dict:
        """Perform a calculation.
        
        Args:
            operation: The operation to perform (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            Result of the calculation
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None,
        }
        
        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}
        
        result = operations[operation](a, b)
        if result is None:
            return {"error": "Cannot divide by zero"}
        
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }
    
    # Create and start the agent
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

