import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, AgentLauncher, Runner, User
from vision_agents.core.utils.examples import get_weather_by_location
from vision_agents.plugins import deepgram, gemini, getstream
from vision_agents.plugins.anam import AnamAvatarPublisher

logger = logging.getLogger(__name__)

load_dotenv()


INSTRUCTIONS = (
    "You're a voice AI assistant. Keep responses short and conversational. "
    "Don't use special characters or formatting. Be friendly and helpful."
)


def setup_llm(model: str = "gemini-3.1-flash-lite-preview") -> gemini.LLM:
    llm = gemini.LLM(model)

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> dict[str, object]:
        return await get_weather_by_location(location)

    return llm


async def create_agent(**kwargs) -> Agent:
    llm = setup_llm()

    avatar = AnamAvatarPublisher()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions=INSTRUCTIONS,
        processors=[avatar],
        llm=llm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    # Have the agent join the call/room
    async with agent.join(call):
        await agent.simple_response("tell me something interesting in a short sentence")

        # run till the call ends
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
