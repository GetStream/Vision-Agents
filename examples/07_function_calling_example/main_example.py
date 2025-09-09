"""
Video call function calling example demonstrating how to use function calling
with an agent connected to a video call.
"""

import asyncio
from uuid import uuid4
from typing import Dict, Any

from dotenv import load_dotenv

from stream_agents.plugins.elevenlabs.tts import TTS as ElevenLabsTTS
from stream_agents.plugins.deepgram.stt import STT as DeepgramSTT
from stream_agents.core.llm.openai_llm import OpenAILLM
from stream_agents.core.agents import Agent
from stream_agents.core.edge import StreamEdge
from stream_agents.core.cli import start_dispatcher
from stream_agents.core.utils import open_demo
from getstream import Stream

load_dotenv()


async def start_agent() -> None:
    """Start an agent with function calling capabilities in a video call."""
    
    # Create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="AI Assistant with Functions")

    # Create the LLM with function calling capabilities
    llm = OpenAILLM(model="gpt-4o-mini")
    
    # Register functions with the LLM
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str) -> Dict[str, Any]:
        """Get the current weather for a location."""
        # Mock weather data - in a real app, you'd call a weather API
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
            "wind_speed": "10 km/h"
        }
        return weather_data
    
    @llm.register_function(description="Calculate mathematical expressions")
    def calculate(expression: str) -> Dict[str, Any]:
        """Calculate mathematical expressions safely."""
        try:
            # Simple safe evaluation - in production, use a proper math parser
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return {
                    "expression": expression,
                    "result": result,
                    "success": True
                }
            else:
                return {
                    "expression": expression,
                    "error": "Invalid characters in expression",
                    "success": False
                }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }
    
    @llm.register_function(description="Get information about a user")
    def get_user_info(user_id: str) -> Dict[str, Any]:
        """Get information about a user by their ID."""
        # Mock user data - in a real app, you'd query a database
        user_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "status": "active",
            "last_login": "2024-01-15",
            "subscription": "premium"
        }
        return user_data
    
    @llm.register_function(description="Convert temperature between units")
    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Convert to Celsius first
        if from_unit == "fahrenheit":
            celsius = (temp - 32) * 5/9
        elif from_unit == "kelvin":
            celsius = temp - 273.15
        elif from_unit == "celsius":
            celsius = temp
        else:
            return {"error": f"Unsupported unit: {from_unit}"}
        
        # Convert from Celsius to target unit
        if to_unit == "fahrenheit":
            result = (celsius * 9/5) + 32
        elif to_unit == "kelvin":
            result = celsius + 273.15
        elif to_unit == "celsius":
            result = celsius
        else:
            return {"error": f"Unsupported unit: {to_unit}"}
        
        return {
            "original_temp": temp,
            "original_unit": from_unit,
            "converted_temp": round(result, 2),
            "converted_unit": to_unit
        }
    
    @llm.register_function(description="Get current time and date")
    def get_current_time() -> Dict[str, Any]:
        """Get the current time and date."""
        from datetime import datetime
        now = datetime.now()
        return {
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "day_of_week": now.strftime("%A"),
            "timezone": "UTC"
        }

    # Create the agent with function calling capabilities
    agent = Agent(
        edge=StreamEdge(),  # low latency edge
        agent_user=agent_user,
        instructions="""You're a helpful AI assistant with access to various functions. 
        You can help users with:
        - Weather information
        - Mathematical calculations
        - User information lookup
        - Temperature conversions
        - Current time and date
        
        When users ask questions that can be answered using your functions, use them automatically.
        Keep responses conversational and helpful. If a function call fails, explain what went wrong.
        Be friendly and engaging in your responses.""",
        llm=llm,
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        processors=[],  # No additional processors needed for this example
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # Send an initial greeting that demonstrates function calling
        await agent.llm.create_response(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Hi! I'm your AI assistant with special powers. I can help you with:
                            - Weather information (try asking about weather in any city)
                            - Math calculations (try asking me to calculate something)
                            - User lookups (try asking about user information)
                            - Temperature conversions (try asking me to convert temperatures)
                            - Current time (ask me what time it is)
                            
                            What would you like to try first?"""
                        }
                    ],
                }
            ]
        )

        # Wait for the call to end
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
