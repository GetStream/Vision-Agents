#!/usr/bin/env python3
"""
Example: Using OpenAI Model with Stream Agents

This example shows how to use the new OpenAI model implementation
with the Agent system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from models import OpenAIModel
from agents import Agent


async def test_openai_model_standalone():
    """Test the OpenAI model standalone."""
    print("🤖 Testing OpenAI Model Standalone")
    print("=" * 50)

    # Create model with environment variables
    model = OpenAIModel(name="gpt-4o-mini")  # Using mini for cost efficiency

    print(f"Model: {model}")
    print(f"Model name: {model.name}")
    print(f"Is async: {model.is_async}")

    # Test simple generation
    try:
        response = await model.generate("What is the capital of France?")
        print("\n💬 Simple generation:")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in simple generation: {e}")

    # Test chat generation
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."},
        ]
        response = await model.generate_chat(messages)
        print("\n💬 Chat generation:")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in chat generation: {e}")

    # Test streaming generation
    try:
        print("\n🌊 Streaming generation:")
        print("Response: ", end="", flush=True)
        async for chunk in model.generate_stream("Tell me a short joke."):
            print(chunk, end="", flush=True)
        print()  # New line after streaming
    except Exception as e:
        print(f"❌ Error in streaming generation: {e}")


async def test_agent_with_openai_model():
    """Test the Agent with OpenAI model."""
    print("\n🤖 Testing Agent with OpenAI Model")
    print("=" * 50)

    # Create OpenAI model
    model = OpenAIModel(
        name="gpt-4o-mini", default_temperature=0.7, default_max_tokens=100
    )

    # Create agent with OpenAI model
    agent = Agent(
        instructions="You are a friendly AI assistant that gives concise, helpful responses.",
        model=model,
        name="OpenAI Assistant",
    )

    print(f"Agent: {agent.name}")
    print(f"Agent model: {agent.model}")
    print(f"Agent bot ID: {agent.bot_id}")

    # Test response generation through agent
    try:
        response = await agent._generate_response(
            "What are the benefits of renewable energy?"
        )
        print("\n💬 Agent response:")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error in agent response: {e}")

    # Test greeting generation
    try:
        greeting = await agent._generate_greeting(2)
        print("\n👋 Agent greeting:")
        print(f"Greeting: {greeting}")
    except Exception as e:
        print(f"❌ Error in agent greeting: {e}")


async def main():
    """Main function to run all examples."""
    load_dotenv()

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in the .env file or environment")
        return

    print("🚀 OpenAI Model Examples")
    print("=" * 50)

    try:
        # Test standalone model
        await test_openai_model_standalone()

        # Test agent with model
        await test_agent_with_openai_model()

        print("\n✅ All examples completed successfully!")

    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    asyncio.run(main())
