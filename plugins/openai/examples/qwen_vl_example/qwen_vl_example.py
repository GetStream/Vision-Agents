import asyncio
import os

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, getstream, openai
from vision_agents.plugins.getstream import CallSessionParticipantJoinedEvent

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = openai.ChatCompletionsVLM(
        model="qwen3.5-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        frame_buffer_seconds=3,
        frame_width=512,
        frame_height=384,
    )
    # Create an agent with video understanding capabilities
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Video Assistant", id="agent"),
        instructions="You're a helpful video AI assistant. Analyze the video frames and respond to user questions about what you see. Keep responses to one sentence. Be concise and direct.",
        llm=llm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        processors=[],
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
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
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
