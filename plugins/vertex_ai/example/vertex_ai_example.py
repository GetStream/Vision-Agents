"""
Simple example demonstrating Vertex AI LLM integration.

Requirements:
- GCP_PROJECT environment variable set
- GCP credentials configured (gcloud auth application-default login)
- Vertex AI API enabled in your project
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

from vision_agents.plugins import vertex_ai

# Load .env file from project root
# When running from project root with: uv run python plugins/vertex_ai/example/vertex_ai_example.py
# the .env file should be in the project root directory
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / ".env"

# Try loading from project root explicitly
if env_file.exists():
    load_dotenv(env_file, override=True)
else:
    # Fallback: search from current working directory upward
    load_dotenv()


async def main():
    # Get configuration from environment
    project = os.getenv("GCP_PROJECT")
    location = os.getenv("GCP_LOCATION", "us-central1")
    
    if not project:
        print("Error: GCP_PROJECT environment variable not set")
        print("\nOptions:")
        print(f"  1. Create a .env file in the project root ({project_root}) with:")
        print("     GCP_PROJECT=your-project-id")
        print("     GCP_LOCATION=us-central1")
        print("\n  2. Or set it as environment variable:")
        print("     export GCP_PROJECT='your-project-id'")
        print(f"\n  Note: Looking for .env at: {env_file}")
        print(f"  .env file exists: {env_file.exists()}")
        return
    
    print(f"Initializing Vertex AI LLM...")
    print(f"Project: {project}")
    print(f"Location: {location}")
    print(f"Model: gemini-1.5-flash\n")
    
    # Initialize LLM
    llm = vertex_ai.LLM(
        model="gemini-1.5-flash",  # Use flash for faster/cheaper testing
        project=project,
        location=location
    )
    
    # Example 1: Simple response
    print("=" * 60)
    print("Example 1: Simple Response")
    print("=" * 60)
    response = await llm.simple_response("Say hello in one sentence!")
    print(f"Response: {response.text}\n")
    
    # Example 2: Streaming response
    print("=" * 60)
    print("Example 2: Streaming Response")
    print("=" * 60)
    chunk_count = 0
    full_text = ""
    
    @llm.events.subscribe
    async def on_chunk(event):
        nonlocal chunk_count, full_text
        chunk_count += 1
        full_text += event.delta
        # Print chunks as they arrive
        print(event.delta, end="", flush=True)
    
    await llm.simple_response("Tell me a short story about a robot (2-3 sentences)")
    await llm.events.wait()  # Wait for all events to process
    
    print(f"\n\nReceived {chunk_count} chunks\n")
    
    # Example 3: Conversation memory
    print("=" * 60)
    print("Example 3: Conversation Memory")
    print("=" * 60)
    llm2 = vertex_ai.LLM(
        model="gemini-1.5-flash",
        project=project,
        location=location
    )
    
    response1 = await llm2.simple_response("My name is Alice and I have 3 cats")
    print(f"Message 1: My name is Alice and I have 3 cats")
    print(f"Response 1: {response1.text}\n")
    
    response2 = await llm2.simple_response("How many paws do my cats have in total?")
    print(f"Message 2: How many paws do my cats have in total?")
    print(f"Response 2: {response2.text}\n")
    
    # Example 4: Using native API
    print("=" * 60)
    print("Example 4: Using Native generate_content API")
    print("=" * 60)
    llm3 = vertex_ai.LLM(
        model="gemini-1.5-flash",
        project=project,
        location=location
    )
    
    response = await llm3.generate_content(message="What is the capital of France?")
    print(f"Response: {response.text}\n")
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
