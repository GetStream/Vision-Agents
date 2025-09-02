"""
Function Calling Example

This example demonstrates how to use the automatic function calling system
with both OpenAI and Gemini Live models. Functions are registered using
decorators and schemas are automatically inferred from function signatures.
"""

import asyncio
import logging
from uuid import uuid4
from typing import List, Optional
from enum import Enum

from dotenv import load_dotenv
from getstream import Stream
from getstream.plugins import DeepgramSTT, ElevenLabsTTS
from stream_agents.turn_detection import FalTurnDetection
from stream_agents.llm import OpenAILLM, GeminiLiveModel
from stream_agents import Agent, StreamEdge, start_dispatcher, open_demo

load_dotenv()


class WeatherCondition(Enum):
    """Enum for weather conditions."""
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    SNOWY = "snowy"


async def create_openai_agent_with_functions():
    """Create an OpenAI agent with function calling capabilities."""
    
    # Create OpenAI LLM
    llm = OpenAILLM(
        name="gpt-4o",
        instructions="You are a helpful AI assistant with access to various tools. Use them when appropriate to help users."
    )
    
    # Register functions using decorators - schemas are automatically inferred!
    
    @llm.function("Get current weather for a location")
    async def get_weather(location: str, unit: str = "celsius") -> str:
        """
        Get weather information for a specific location.
        
        Args:
            location: The city or location to get weather for
            unit: Temperature unit (celsius or fahrenheit)
        """
        # Simulate API call
        await asyncio.sleep(0.1)  # Simulate network delay
        temp = 22 if unit == "celsius" else 72
        return f"Weather in {location}: {temp}°{unit[0].upper()}, {WeatherCondition.SUNNY.value}"
    
    @llm.function("Send a message to a user")
    def send_message(user_id: str, message: str, priority: str = "normal") -> str:
        """
        Send a message to a specific user.
        
        Args:
            user_id: The ID of the user to send the message to
            message: The message content to send
            priority: Message priority (low, normal, high)
        """
        print(f"📨 Sending {priority} priority message to {user_id}: {message}")
        return f"Message sent to {user_id} with {priority} priority"
    
    @llm.function("Get user information")
    async def get_user_info(user_id: str) -> dict:
        """
        Retrieve information about a user.
        
        Args:
            user_id: The ID of the user to get information for
        """
        # Simulate database lookup
        await asyncio.sleep(0.05)
        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"{user_id}@example.com",
            "status": "active"
        }
    
    @llm.function("Calculate mathematical expressions")
    def calculate(expression: str) -> str:
        """
        Safely evaluate mathematical expressions.
        
        Args:
            expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        """
        try:
            # Simple safe evaluation (in production, use a proper math parser)
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    @llm.function("Get list of available commands")
    def list_commands() -> List[str]:
        """Get a list of all available commands."""
        return [
            "get_weather - Get weather for a location",
            "send_message - Send a message to a user", 
            "get_user_info - Get user information",
            "calculate - Calculate mathematical expressions",
            "list_commands - List all available commands"
        ]
    
    # Create the agent
    agent = Agent(
        edge=StreamEdge(),
        llm=llm,
        tts=ElevenLabsTTS(),
        stt=DeepgramSTT(),
        turn_detection=FalTurnDetection(),
    )
    
    return agent


async def create_gemini_agent_with_functions():
    """Create a Gemini Live agent with function calling capabilities."""
    
    # Create Gemini Live LLM
    llm = GeminiLiveModel(
        model="gemini-2.5-flash-preview-native-audio-dialog",
        instructions="You are a helpful AI assistant with access to various tools. Use them when appropriate to help users."
    )
    
    # Register the same functions for Gemini
    @llm.function("Get current weather for a location")
    async def get_weather(location: str, unit: str = "celsius") -> str:
        """Get weather information for a specific location."""
        await asyncio.sleep(0.1)
        temp = 22 if unit == "celsius" else 72
        return f"Weather in {location}: {temp}°{unit[0].upper()}, {WeatherCondition.SUNNY.value}"
    
    @llm.function("Send a message to a user")
    def send_message(user_id: str, message: str, priority: str = "normal") -> str:
        """Send a message to a specific user."""
        print(f"📨 Sending {priority} priority message to {user_id}: {message}")
        return f"Message sent to {user_id} with {priority} priority"
    
    @llm.function("Calculate mathematical expressions")
    def calculate(expression: str) -> str:
        """Safely evaluate mathematical expressions."""
        try:
            allowed_chars = set('0123456789+-*/()., ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create the agent (STS mode - no separate STT/TTS needed)
    agent = Agent(
        edge=StreamEdge(),
        llm=llm,
    )
    
    return agent


async def test_function_calling():
    """Test function calling with a simple conversation."""
    
    # Create OpenAI agent
    llm = OpenAILLM(
        name="gpt-4o",
        instructions="You are a helpful assistant. Use the available functions when appropriate."
    )
    
    @llm.function("Get weather information")
    async def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: 25°C, sunny"
    
    @llm.function("Send notification")
    def send_notification(message: str, urgent: bool = False) -> str:
        """Send a notification message."""
        prefix = "🚨 URGENT: " if urgent else "📢 "
        print(f"{prefix}{message}")
        return f"Notification sent: {message}"
    
    # Test the functions
    print("🧪 Testing function calling...")
    
    # Test 1: Weather query
    response = await llm.generate("What's the weather like in Tokyo?")
    print(f"Response: {response}")
    
    # Test 2: Notification
    response = await llm.generate("Send an urgent notification that the server is down")
    print(f"Response: {response}")
    
    # Test 3: List available functions
    print(f"Available functions: {llm.get_available_functions()}")
    for func_name in llm.get_available_functions():
        info = llm.get_function_info(func_name)
        print(f"  - {func_name}: {info['description']}")


async def main():
    """Main function to run the function calling example."""
    
    print("🚀 Function Calling Example")
    print("=" * 50)
    
    # Test function calling first
    await test_function_calling()
    
    print("\n" + "=" * 50)
    print("🎥 Starting video call with function calling...")
    
    # Create Stream client
    client = Stream.from_env()
    agent_user = client.create_user(name="AI Assistant with Functions")
    
    # Choose which agent to use
    use_gemini = False  # Set to True to use Gemini Live instead
    
    if use_gemini:
        print("Using Gemini Live with function calling...")
        agent = await create_gemini_agent_with_functions()
    else:
        print("Using OpenAI with function calling...")
        agent = await create_openai_agent_with_functions()
    
    # Create a call
    call = client.video.call("default", str(uuid4()))
    
    # Open the demo UI
    open_demo(call)
    
    # Have the agent join the call
    with await agent.join(call):
        # Send initial greeting
        await agent.create_response("Hello! I'm an AI assistant with access to various tools. You can ask me to check the weather, send messages, calculate math, or get user information. What would you like me to help you with?")
        
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_dispatcher(main))
