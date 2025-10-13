"""
Simple Gemini function calling example.
"""

import asyncio
from dotenv import load_dotenv
from vision_agents.plugins import gemini

load_dotenv()


async def main():
    """Run a simple Gemini function calling example."""

    # Create the LLM
    llm = gemini.LLM("gemini-2.0-flash")

    # Register functions
    @llm.register_function(description="Get current weather for a location")
    def get_weather(location: str):
        """Get the current weather for a location."""
        return {
            "location": location,
            "temperature": "22°C",
            "condition": "Sunny",
            "humidity": "65%",
        }

    @llm.register_function(description="Calculate the sum of two numbers")
    def calculate_sum(a: int, b: int):
        """Calculate the sum of two numbers."""
        return a + b

    # Test queries
    queries = [
        "What's the weather like in New York?",
        "Calculate 15 + 27 for me",
        "What's the weather in London and calculate 100 + 200?",
    ]

    print("Gemini Function Calling Example")
    print("=" * 50)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)

        response = await llm.send_message(message=query)

        print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
