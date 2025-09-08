"""
Gemini function calling example demonstrating how to use function calling
with Google's Gemini LLM.
"""

import asyncio
import os
from typing import Dict, Any

from stream_agents.core.llm import GeminiLLM


async def main():
    """Main function to demonstrate Gemini function calling."""
    
    # Initialize Gemini LLM
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Please set GOOGLE_API_KEY environment variable")
        return
    
    llm = GeminiLLM(model="gemini-pro", api_key=api_key)
    
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
    
    @llm.register_function(description="Convert temperature between Celsius and Fahrenheit")
    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Convert temperature between Celsius and Fahrenheit."""
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            converted = (temp * 9/5) + 32
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            converted = (temp - 32) * 5/9
        else:
            raise ValueError(f"Unsupported conversion from {from_unit} to {to_unit}")
        
        return {
            "original_temp": temp,
            "original_unit": from_unit,
            "converted_temp": round(converted, 2),
            "converted_unit": to_unit
        }
    
    print("Gemini Function Calling Example")
    print("=" * 50)
    
    # Show available functions
    print("\nAvailable functions:")
    for schema in llm.get_available_functions():
        print(f"- {schema['name']}: {schema.get('description', 'No description')}")
    
    print("\n" + "=" * 50)
    print("Testing function calls:")
    print("=" * 50)
    
    # Test different queries that should trigger function calls
    test_queries = [
        "What's the weather like in New York?",
        "Calculate 15 + 27 for me",
        "Get information about user '12345'",
        "What's the weather in London and calculate 100 + 200?",
        "Convert 25 degrees Celsius to Fahrenheit",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        try:
            response = await llm.simple_response(query)
            print(f"Response: {response.text}")
            
            # Check if there were any function calls
            if hasattr(response, 'original') and hasattr(response.original, 'candidates'):
                for candidate in response.original.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                print(f"Function call made: {part.function_call.name}({part.function_call.args})")
                        
        except Exception as e:
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    asyncio.run(main())
