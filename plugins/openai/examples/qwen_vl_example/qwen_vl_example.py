import asyncio

from dotenv import load_dotenv
from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.events import CallSessionParticipantJoinedEvent
from vision_agents.plugins import deepgram, elevenlabs, getstream, openai

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    # Initialize the Baseten VLM
    llm = openai.ChatCompletionsVLM(model="qwen3vl")

    # Create an agent with video understanding capabilities
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Video Assistant", id="agent"),
        instructions="You're a helpful video AI assistant. Analyze the video frames and respond to user questions about what you see.",
        llm=llm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(2)
            await agent.simple_response("Describe what you currently see")

    async with agent.join(call):
        await agent.edge.open_demo(call)
        # The agent will automatically process video frames and respond to user input
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
