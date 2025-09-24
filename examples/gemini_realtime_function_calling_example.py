#!/usr/bin/env python3
"""
Example demonstrating function calling with Gemini Realtime.

This example shows how to:
1. Register functions with the Gemini Realtime class
2. Use function calling in real-time conversations
3. Handle MCP integration

The Gemini Live API supports function calling, so this example should work!
"""

import asyncio
import logging
import os
from dotenv import load_dotenv
from stream_agents.plugins.gemini import Realtime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main example function."""
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables")
        return
    
    # Create Gemini Realtime instance
    realtime = Realtime(
        model="gemini-2.5-flash-native-audio-preview-09-2025",
        api_key=api_key
    )
    
    # Register some example functions
    @realtime.register_function(description="Get current weather for a location")
    def get_weather(location: str) -> dict:
        """Get the current weather for a location."""
        # In a real implementation, this would call a weather API
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
    
    @realtime.register_function(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @realtime.register_function(description="Get user information")
    def get_user_info(user_id: str) -> dict:
        """Get information about a user."""
        # In a real implementation, this would query a database
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "last_login": "2024-01-15"
        }
    
    try:
        # Connect to Gemini Live
        await realtime.connect()
        logger.info("Connected to Gemini Live with function calling support")
        
        # Send a text message that might trigger function calls
        await realtime.simple_response(
            "What's the weather like in New York and what's 15 + 27?"
        )
        
        # The realtime class will now:
        # 1. Detect function calls in the response
        # 2. Execute the functions using the existing tool execution infrastructure
        # 3. Send the results back to Gemini Live
        # 4. Continue the conversation with the function results
        
        logger.info("Function calling example completed")
        
    except Exception as e:
        logger.error(f"Error in function calling example: {e}")
    
    finally:
        # Clean up
        await realtime.close()

if __name__ == "__main__":
    print("Gemini Realtime Function Calling Example")
    print("=" * 50)
    print("This example demonstrates function calling with Gemini Realtime.")
    print("The Gemini Live API supports function calling, so this should work!")
    print("=" * 50)
    
    # Run the example
    asyncio.run(main())
