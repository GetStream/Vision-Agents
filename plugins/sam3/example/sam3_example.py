import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, gemini, sam3

logger = logging.getLogger(__name__)

load_dotenv()

"""
SAM3 Video Segmentation Example

This example demonstrates real-time video segmentation using Meta's SAM 3 model
with dynamic prompt changing via function calls.

The AI agent can change what's being segmented by calling the processor's 
change_prompt function.
"""


async def create_agent(**kwargs) -> Agent:
    # Create SAM3 processor for video segmentation
    sam3_processor = sam3.VideoSegmentationProcessor(
        text_prompt="person",  # Initial segmentation target
        threshold=0.5,         # Confidence threshold
        mask_threshold=0.5,    # Mask binarization threshold
        fps=30,                # Frame processing rate
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="SAM3 Segmentation Assistant"),
        instructions="Read @sam3_instructions.md",
        llm=gemini.Realtime(fps=3),  # Share video with gemini
        processors=[sam3_processor],  # Add SAM3 segmentation processor
    )

    # Register the change_prompt function so the AI can call it
    @agent.llm.register_function(description="Change what object/concept to segment in the video. Finds ALL instances matching the prompt.")
    async def change_prompt(prompt: str) -> dict:
        """
        Change the segmentation prompt dynamically.
        
        Args:
            prompt: Text description of what to segment (e.g., "person", "car", "dog", "basketball")
        
        Returns:
            Status message indicating the prompt was changed
        """
        return await sam3_processor.change_prompt(prompt)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    with await agent.join(call):
        await agent.llm.simple_response(
            text="Hello! I'm your video segmentation assistant. I can segment objects in your video feed. "
                 "Just tell me what you'd like me to segment - like 'segment people' or 'find all the cars'."
        )
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
