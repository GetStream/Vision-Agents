"""
Function calling example demonstrating how to use the function calling capabilities
with different LLM providers.
"""

import asyncio
import os
from typing import Dict, Any

from stream_agents.core.llm import OpenAILLM


async def main():
    """Main function to demonstrate function calling."""
    
    # Initialize LLM (you can switch between different providers)
    # Make sure to set your API keys in environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    llm = OpenAILLM(model="gpt-4", api_key=api_key)
    
    # Register functions with the LLM using decorator syntax
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str) -> Dict[str, Any]:
        """Get the current weather for a location."""
        # Mock weather data
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
        return weather_data
    
    @llm.register_function(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    @llm.register_function(description="Get user information by ID")
    def get_user_info(user_id: str) -> Dict[str, Any]:
        """Get information about a user by their ID."""
        # Mock user data
        user_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "status": "active"
        }
        return user_data
    
    print("Available functions:")
    for schema in llm.get_available_functions():
        print(f"- {schema['name']}: {schema.get('description', 'No description')}")
    
    print("\n" + "="*50)
    print("Function Calling Example")
    print("="*50)
    
    # Test different queries that should trigger function calls
    test_queries = [
        "What's the weather like in New York?",
        "Calculate 15 + 27 for me",
        "Get information about user '12345'",
        "What's the weather in London and calculate 100 + 200?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            response = await llm.simple_response(query)
            print(f"Response: {response.text}")
            
            # Check if there were any function calls
            if hasattr(response, 'original') and hasattr(response.original, 'tool_calls'):
                print("Function calls made:")
                for tool_call in response.original.tool_calls:
                    print(f"  - {tool_call.function.name}({tool_call.function.arguments})")
                    
        except Exception as e:
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())
