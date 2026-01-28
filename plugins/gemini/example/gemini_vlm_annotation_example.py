

import logging
from typing import Any, Dict

from dotenv import load_dotenv

from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.processors import AnnotationProcessor
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream
from google.genai import types


logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    annotation_processor = AnnotationProcessor(
        fps=30,
        annotation_duration=10.0,  # Show annotations for 10 seconds
        default_style={
            "color": (0, 255, 0),
            "thickness": 3,
            "font_scale": 0.7,
            "font_thickness": 2,
        },
    )


    llm = gemini.VLM(
        model="gemini-3-flash-preview",
        fps=0,
        frame_format="jpeg",
        frame_width=800,
        frame_height=600,
        enable_structured_annotations=True,
        config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    ),
    )


    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Vision Agent", id="agent"),
        instructions=(
            "You're a helpful AI assistant with vision and annotation capabilities. "
            "You can see the video feed and mark objects by circling, boxing, or highlighting them. "
            "\n\n"
            "When the user asks you to circle, mark, highlight, or draw a box around something:\n"
            "1. Analyze the video frames to locate the object\n"
            "2. Return JSON with bounding box coordinates in the format shown in your system instructions\n"
            "3. Add a friendly confirmation message after the JSON\n"
            "\n"
            "Example request: 'Circle my face'\n"
            "Your response: Return JSON with face coordinates, then say 'I've circled your face!'\n"
            "\n"
            "Keep your responses conversational and concise. "
            "You can also perform code execution for complex asks such as if the user asks to identify the number of fingers or calculate something"
            "Don't use special characters or emoji in your speech."
        ),
        processors=[annotation_processor],
        llm=llm,
        tts=elevenlabs.TTS(model_id="eleven_flash_v2_5"),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    # Register custom functions that can be called by the LLM
    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        """Get weather information for the given location."""
        return await get_weather_by_location(location)

    @llm.register_function(
        description="Count specific objects visible in the current video frame"
    )
    async def count_objects(object_type: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "object_type": object_type,
            "count": "N/A - implement with CV library",
            "message": f"This would count {object_type} using computer vision",
        }

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        # Send an initial greeting
        await agent.simple_response(
            "Greet the user warmly and let them know you can see their video feed. "
            "Explain that you can mark or circle objects when they ask you to. "
            "For example, if they say 'circle my face', you'll draw a box around their face on the video. "
            "Keep it brief and friendly."
        )

        await agent.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # Run the agent with CLI interface
    print("\n" + "=" * 80)
    print("Gemini VLM with Visual Annotations Example")
    print("=" * 80)
    print("\nThis agent can see your video and draw annotations on it!")
    print("\nTry saying:")
    print("  - 'Circle my face'")
    print("  - 'Draw a box around me'")
    print("  - 'Mark what you see'")
    print("  - 'Highlight my hands'")
    print("\nThe annotations will appear on your video feed for 10 seconds.")
    print("=" * 80 + "\n")

    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
