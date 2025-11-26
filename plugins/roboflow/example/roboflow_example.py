import asyncio
import os
from uuid import uuid4
from dotenv import load_dotenv

from vision_agents.core import User, Agent
from vision_agents.plugins import cartesia, deepgram, getstream, gemini, vogent, roboflow

load_dotenv()


async def start_agent() -> None:
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

    # TODO: This is broken
    # Create Roboflow processor
    roboflow_processor = roboflow.RoboflowDetectionProcessor(
        model_id="soccer-players-ckbru/15",
        # api_key will be read from ROBOFLOW_API_KEY env var
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
    
    # Have the agent join the call
    with await agent.join(call):
        # Open the demo UI
        await agent.edge.open_demo(call)
        
        # Greet and explain capabilities
        await agent.simple_response(
            "Hello! I can see your video feed and detect objects. "
            "Try showing me different objects and ask what I can see!"
        )
        
        # Run until the call ends
        await agent.finish()


if __name__ == "__main__":
    asyncio.run(start_agent())

