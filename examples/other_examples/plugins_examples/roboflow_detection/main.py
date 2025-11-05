#!/usr/bin/env python3
"""
Example: Real-time Object Detection with Roboflow

This example demonstrates how to:
1. Create an Agent with Roboflow object detection
2. Join a Stream video call
3. Process video frames with your trained Roboflow model
4. Annotate frames with detected objects
5. Interact with users about what the agent sees

Usage:
    python main.py

Requirements:
    - Create a .env file with your credentials (see env.example)
    - Train a model on Roboflow or use an existing one
    - Install dependencies: uv sync
"""

import asyncio
import os
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import cartesia, deepgram, getstream, gemini, vogent, roboflow

load_dotenv()


async def main() -> None:
    """
    Example agent using Roboflow for object detection.
    
    Required environment variables:
    - ROBOFLOW_API_KEY: Your Roboflow API key
    - ROBOFLOW_WORKSPACE: Your Roboflow workspace ID
    - ROBOFLOW_PROJECT: Your Roboflow project ID
    - ROBOFLOW_VERSION: Model version number (default: 1)
    """
    
    # Get Roboflow configuration from environment
    workspace_id = os.getenv("ROBOFLOW_WORKSPACE")
    project_id = os.getenv("ROBOFLOW_PROJECT")
    model_version = int(os.getenv("ROBOFLOW_VERSION", "1"))
    
    if not workspace_id or not project_id:
        raise ValueError(
            "Missing Roboflow configuration. Set ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT env vars.\n"
            "Get these from your Roboflow project URL: https://app.roboflow.com/YOUR_WORKSPACE/YOUR_PROJECT"
        )
    
    print("🔍 Roboflow Object Detection Example")
    print("=" * 50)
    print(f"📦 Using model: {workspace_id}/{project_id} v{model_version}")
    print()
    
    # Create Roboflow processor
    roboflow_processor = roboflow.RoboflowDetectionProcessor(
        # api_key will be read from ROBOFLOW_API_KEY env var
        workspace_id=workspace_id,
        project_id=project_id,
        model_version=model_version,
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
        turn_detection=vogent.TurnDetection(),
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

