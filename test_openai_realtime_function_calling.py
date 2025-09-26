#!/usr/bin/env python3
"""
Test script for OpenAI Realtime function calling integration.

This script demonstrates how to use the OpenAI Realtime class with function calling
and MCP support. It creates a simple agent that can call functions during real-time
conversations.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from stream_agents.plugins.openai.openai_realtime import Realtime
from stream_agents.plugins import getstream
from stream_agents.core.edge.types import User

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openai_realtime_function_calling():
    """Test OpenAI Realtime with function calling capabilities."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not found!")
        return
    
    if not os.getenv("STREAM_API_KEY"):
        logger.error("STREAM_API_KEY environment variable not found!")
        return
    
    # Create OpenAI Realtime LLM
    llm = Realtime(
        model="gpt-realtime",
        voice="marin"
    )
    
    # Register some test functions
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str) -> dict:
        """Get the current weather for a location."""
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
    
    @llm.register_function(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @llm.register_function(description="Get current time")
    def get_current_time() -> str:
        """Get the current time."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create edge and agent user
    edge = getstream.Edge()
    agent_user = User(name="OpenAI Function Calling Assistant", id="openai-assistant")
    
    logger.info("Created OpenAI Realtime LLM with function calling support")
    logger.info(f"Available functions: {[f['name'] for f in llm.get_available_functions()]}")
    
    try:
        # Connect to OpenAI realtime
        await llm.connect()
        logger.info("✅ Connected to OpenAI Realtime API")
        
        # Test simple response (this should work without function calling)
        logger.info("Testing simple response...")
        await llm.simple_response("Hello! I'm your AI assistant with function calling capabilities.")
        
        # Test function calling via text input
        logger.info("Testing function calling...")
        await llm.simple_response("What's the weather like in New York?")
        
        # Wait a bit for the response
        await asyncio.sleep(2)
        
        # Test another function call
        await llm.simple_response("Calculate 15 + 27 for me")
        
        # Wait a bit for the response
        await asyncio.sleep(2)
        
        # Test multiple function calls
        await llm.simple_response("What time is it and what's the weather in London?")
        
        # Wait for responses
        await asyncio.sleep(3)
        
        logger.info("✅ Function calling tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        await llm.close()
        logger.info("Closed OpenAI Realtime connection")


if __name__ == "__main__":
    asyncio.run(test_openai_realtime_function_calling())
