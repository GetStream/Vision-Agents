import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import decart, deepgram, elevenlabs, getstream, openai

logger = logging.getLogger(__name__)

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    processor = decart.RestylingProcessor(
        initial_prompt="A cute animated movie with vibrant colours", model="mirage_v2"
    )
    llm = openai.LLM(model="gpt-4o-mini")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Story teller", id="agent"),
        instructions="You are a story teller. You will tell a short story to the user. You will use the Decart processor to change the style of the video and user's background. You can embed audio tags in your responses for added effect Emotional tone: [EXCITED], [NERVOUS], [FRUSTRATED], [TIRED] Reactions: [GASP], [SIGH], [LAUGHS], [GULPS] Volume & energy: [WHISPERING], [SHOUTING], [QUIETLY], [LOUDLY] Pacing & rhythm: [PAUSES], [STAMMERS], [RUSHED]",
        llm=llm,
        tts=elevenlabs.TTS(voice_id="N2lVS1w4EtoT3dr4eOWO"),
        stt=deepgram.STT(),
        processors=[processor],
    )

    @llm.register_function(
        description="This function changes the prompt of the Decart processor which in turn changes the style of the video and user's background"
    )
    async def change_prompt(prompt: str) -> str:
        await processor.update_prompt(prompt)
        return f"Prompt changed to {prompt}"

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Agent...")

    # Have the agent join the call/room
    async with agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
