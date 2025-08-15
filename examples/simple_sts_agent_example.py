#!/usr/bin/env python3
"""
Simple STS Agent Example

This example demonstrates the simplest way to create and join a call using the Agent class
with Google's Gemini Live API in native audio mode (STS - Speech-to-Speech). This is a minimal example 
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

# Import the Agent class and Gemini Live model
from agents.agents2 import Agent
from models.gemini_sts import GeminiLiveModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main() -> None:
    """Create a simple STS agent and join a call."""
    
    load_dotenv()
    
    # Initialize Stream client
    client: Stream = Stream.from_env()

    # Create the simplest possible STS agent
    agent_user = UserRequest(id=str(uuid4()), name="My AI Assistant")
    
    # Create Gemini Live STS model with native audio mode
    sts_model = GeminiLiveModel(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash-preview-native-audio-dialog",  # Native audio model
        instructions=(
            "You are an English-speaking voice AI assistant. You MUST always respond in English only. "
            "Keep responses short and conversational. Don't use special characters or formatting. "
            "Be friendly and helpful. Speak naturally and respond to interruptions appropriately. "
            "Never speak in any language other than English."
        ),
        response_modalities=["AUDIO"],  # Use audio output
    )

    # Create agent with Gemini Live STS model (no separate STT/TTS needed)
    agent = Agent(
        llm=sts_model,  # Pass Gemini Live model as the LLM
        agent_user=agent_user,

    )

    client.upsert_users(UserRequest(id=agent_user.id, name=agent_user.name))

    try:
        # Join the call - this is the main functionality we're demonstrating
        call = client.video.call("default", str(uuid4()))
        open_demo(client, call.id)
        await agent.join(call)
        
        # Keep the agent running
        logging.info("🤖 Gemini Live Agent has joined the call. Press Ctrl+C to exit.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("👋 Shutting down Gemini Live agent...")
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
