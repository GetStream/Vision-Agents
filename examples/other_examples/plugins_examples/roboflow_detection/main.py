#!/usr/bin/env python3
"""
Example: Real-time Object Detection with Roboflow

This example demonstrates how to:
1. Create an Agent with Roboflow object detection
2. Join a Stream video call
3. Process video frames with a Roboflow model
4. Annotate frames with detected objects
5. Interact with users about what the agent sees

Usage:
    python main.py

Requirements:
    - Create a .env file with your credentials (see env.example)
    - Either use your own trained model or a public Roboflow Universe model
    - Install dependencies: uv sync
"""

import asyncio
import os
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import cartesia, deepgram, getstream, gemini, roboflow

load_dotenv()


async def main() -> None:
    """
    Example agent using Roboflow Universe models for object detection.
    
    Find models at: https://universe.roboflow.com
    Format: ROBOFLOW_MODEL_ID=workspace/project or workspace/project/version
    
    Examples:
    - ROBOFLOW_MODEL_ID=roboflow-100/aerial-spheres
    - ROBOFLOW_MODEL_ID=roboflow-100/aerial-spheres/2
    - ROBOFLOW_MODEL_ID=microsoft/coco
    """
    
    # Get Roboflow configuration from environment
    model_id = os.getenv("ROBOFLOW_MODEL_ID")
    version = os.getenv("ROBOFLOW_VERSION")
    
    if not model_id:
        raise ValueError(
            "Missing ROBOFLOW_MODEL_ID.\n"
            "Format: workspace/project or workspace/project/version\n"
            "Example: ROBOFLOW_MODEL_ID=roboflow-100/aerial-spheres\n"
            "Find models at: https://universe.roboflow.com"
        )
    
    print("🔍 Roboflow Object Detection Example")
    print("=" * 50)
    print(f"📦 Using model: {model_id}" + (f" v{version}" if version else ""))
    print()
    
    # Create Roboflow processor
    # API key will be read from ROBOFLOW_API_KEY env var if needed
    roboflow_processor = roboflow.RoboflowDetectionProcessor(
        model_id=model_id,
        version=int(version) if version else None,
        conf_threshold=40,  # 40% confidence threshold
        fps=5,  # Process 5 frames per second (API-friendly)
    )
    
    # Create agent with Roboflow processor
    llm = gemini.LLM("gemini-2.0-flash")
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="Vision AI Assistant", 
            id="agent"
        ),
        instructions=(
            "You're a vision AI assistant with object detection capabilities. "
            "You can see what's in the video feed through the Roboflow detector. "
            "Describe what objects you detect in a natural, conversational way. "
            "Keep responses short and friendly."
        ),
        processors=[roboflow_processor],
        llm=llm,
        tts=cartesia.TTS(),
        stt=deepgram.STT(),
    )
    
    # Create a call
    call = agent.edge.client.video.call("default", str(uuid4()))
    
    print("🤖 Agent created successfully!")
    print()
    
    try:
        # Have the agent join the call
        with await agent.join(call):
            print("✅ Agent joined the call")
            
            # Open the demo UI
            await agent.edge.open_demo(call)
            print("🌐 Demo UI opened in your browser")
            print()
            print("👉 Enable your camera and show objects to test detection")
            print("👉 Ask the agent 'What do you see?' to hear it describe the objects")
            print()
            print("Press Ctrl+C to stop")
            print()
            
            # Greet and explain capabilities
            await agent.simple_response(
                "Hello! I can see your video feed and detect objects. "
                "Try showing me different objects and ask what I can see!"
            )
            
            # Run until the call ends
            await agent.finish()
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopping agent...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    print("🎥 Stream + Roboflow Real-time Object Detection")
    print("=" * 50)
    print()
    asyncio.run(main())

