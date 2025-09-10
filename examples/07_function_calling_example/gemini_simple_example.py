"""
Simple Gemini example (without function calling for now).
"""

import asyncio
from dotenv import load_dotenv
from stream_agents.core.llm.gemini_llm import GeminiLLM

load_dotenv()


async def main():
    """Run a simple Gemini example."""
    
    # Create the LLM
    llm = GeminiLLM(model="gemini-1.5-flash")
    
    # Test queries
    queries = [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Tell me a joke",
        "What's the weather like today?",
        "Can you help me with math?"
    ]
    
    print("Gemini Simple Example")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        response = await llm.send_message(
            message=query
        )
        
        print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
