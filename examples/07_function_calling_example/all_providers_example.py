"""
Comprehensive function calling example demonstrating how to use function calling
with all supported LLM providers (OpenAI, Claude, and Gemini).
"""

import asyncio
import os
from typing import Dict, Any

from stream_agents.core.llm import OpenAILLM, ClaudeLLM, GeminiLLM


async def test_provider(llm, provider_name: str, query: str):
    """Test a specific LLM provider with a query."""
    print(f"\n{provider_name} Response:")
    print("-" * 30)
    
    try:
        response = await llm.simple_response(query)
        print(f"Response: {response.text}")
        
        # Check for function calls based on provider
        if provider_name == "OpenAI" and hasattr(response, 'original') and hasattr(response.original, 'choices'):
            for choice in response.original.choices:
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        print(f"Function call: {tool_call.function.name}({tool_call.function.arguments})")
        
        elif provider_name == "Claude" and hasattr(response, 'original') and hasattr(response.original, 'content'):
            for content_block in response.original.content:
                if content_block.type == "tool_use":
                    print(f"Function call: {content_block.name}({content_block.input})")
        
        elif provider_name == "Gemini" and hasattr(response, 'original') and hasattr(response.original, 'candidates'):
            for candidate in response.original.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            print(f"Function call: {part.function_call.name}({part.function_call.args})")
                            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Main function to demonstrate function calling across all providers."""
    
    print("Function Calling Across All LLM Providers")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What's the weather like in New York?",
        "Calculate 15 + 27 for me",
        "Get information about user '12345'",
        "Convert 25 degrees Celsius to Fahrenheit",
    ]
    
    # Initialize all LLM providers
    providers = []
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_llm = OpenAILLM(model="gpt-4", api_key=openai_key)
        providers.append(("OpenAI", openai_llm))
    else:
        print("Skipping OpenAI - OPENAI_API_KEY not set")
    
    # Claude
    claude_key = os.getenv("ANTHROPIC_API_KEY")
    if claude_key:
        claude_llm = ClaudeLLM(model="claude-3-sonnet-20240229", api_key=claude_key)
        providers.append(("Claude", claude_llm))
    else:
        print("Skipping Claude - ANTHROPIC_API_KEY not set")
    
    # Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        gemini_llm = GeminiLLM(model="gemini-pro", api_key=gemini_key)
        providers.append(("Gemini", gemini_llm))
    else:
        print("Skipping Gemini - GOOGLE_API_KEY not set")
    
    if not providers:
        print("No API keys found. Please set at least one of:")
        print("- OPENAI_API_KEY")
        print("- ANTHROPIC_API_KEY") 
        print("- GOOGLE_API_KEY")
        return
    
    # Register functions with all providers
    def get_weather(location: str) -> Dict[str, Any]:
        """Get the current weather for a location."""
        weather_data = {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%"
        }
        return weather_data
    
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        return a + b
    
    def get_user_info(user_id: str) -> Dict[str, Any]:
        """Get information about a user by their ID."""
        user_data = {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "status": "active"
        }
        return user_data
    
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
    
    # Register functions with all providers
    for provider_name, llm in providers:
        llm.register_function(description="Get current weather for a location")(get_weather)
        llm.register_function(description="Calculate the sum of two numbers")(calculate_sum)
        llm.register_function(description="Get user information by ID")(get_user_info)
        llm.register_function(description="Convert temperature between Celsius and Fahrenheit")(convert_temperature)
    
    # Test each query with all available providers
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        for provider_name, llm in providers:
            await test_provider(llm, provider_name, query)
        
        print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())
