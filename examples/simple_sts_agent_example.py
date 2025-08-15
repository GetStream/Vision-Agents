#!/usr/bin/env python3
"""
Simple STS Agent Example

This example demonstrates the simplest way to create and join a call using the Agent class
with OpenAI's realtime voice model (STS - Speech-to-Speech). This is a minimal example 
that shows just the core functionality without separate STT/TTS services.
"""

import asyncio
import logging
import os
import sys
import webbrowser
from pathlib import Path
from uuid import uuid4
from urllib.parse import urlencode

from getstream import Stream
from getstream.video.rtc.utils import open_browser

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from getstream.models import UserRequest, User
from getstream.stream import Stream

from utils import open_demo

# Import the Agent class and OpenAI STS model
from agents.agents2 import Agent
from models.openai_sts import OpenAIRealtimeModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main() -> None:
    """Create a simple STS agent and join a call."""
    
    load_dotenv()
    
    # Initialize Stream client
    client: Stream = Stream.from_env()

    # Create the simplest possible STS agent
    agent_user = UserRequest(id=str(uuid4()), name="My AI Assistant")
    
    # Create OpenAI Realtime STS model
    sts_model = OpenAIRealtimeModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-realtime-preview",
        voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
        instructions=(
            "You are an English-speaking voice AI assistant. You MUST always respond in English only. "
            "Keep responses short and conversational. Don't use special characters or formatting. "
            "Be friendly and helpful. Speak naturally and respond to interruptions appropriately. "
            "Never speak in any language other than English."
        ),
        turn_detection={
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200,
        },
        input_audio_transcription={"model": "whisper-1", "language": "en"},
        # Additional session parameters to ensure English output
        output_audio_format="pcm16",
    )

    # Create agent with STS model (no separate STT/TTS needed)
    agent = Agent(
        llm=sts_model,  # Pass STS model as the LLM
        agent_user=agent_user,
    )

    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        open_demo(client, call.id)
        await agent.join(call)
        
        # Keep the agent running
        logging.info("🤖 STS Agent has joined the call. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("👋 Shutting down STS agent...")
    except Exception as e:
        logging.error("❌ Error: %s", e)
    finally:
        # Clean up agent resources
        if 'agent' in locals():
            try:
                await agent.close()
                logging.info("✅ Agent cleanup completed")
            except Exception as e:
                logging.error(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    asyncio.run(main())
