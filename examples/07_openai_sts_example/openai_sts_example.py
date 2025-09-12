"""
OpenAI STS (Speech-to-Speech) Example

This example demonstrates using OpenAI's Realtime API for speech-to-speech conversation.
The agent uses WebRTC to establish a peer connection with OpenAI's servers, enabling
real-time bidirectional audio streaming.
"""

import asyncio
import logging
from uuid import uuid4
from dotenv import load_dotenv

from getstream import Stream

from stream_agents import open_demo
from stream_agents.agents import Agent
from stream_agents.llm import OpenAIRealtimeLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the OpenAI STS agent"""
    # Load environment variables
    load_dotenv()

    # Initialize Stream client
    client = Stream.from_env()

    # Create a unique call ID
    call_id = str(uuid4())

    # Get or create the call
    call = client.video.call("default", call_id)

    # Open browser for human to join
    open_demo(call)

    llm = OpenAIRealtimeLLM()

    # Create agent with OpenAI STS
    agent = Agent(
        llm=llm,
        instructions=(
            "You are a helpful AI assistant engaged in a voice conversation. "
            "Keep your responses natural, conversational, and concise. "
            "You can interrupt and be interrupted naturally, just like in a real conversation."
        ),
        agent_user=client.create_user(name="My happy AI friend"),
    )

    logger.info("🤖 Starting OpenAI STS Agent...")

    with await agent.join(call):
        img_url = "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        await agent.llm.create_response(
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Tell me a short poem about this image",
                        },
                        {"type": "input_image", "image_url": f"{img_url}"},
                    ],
                }
            ]
        )
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(main())
