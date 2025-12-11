import asyncio

from dotenv import load_dotenv
from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.core.events import CallSessionParticipantJoinedEvent
from vision_agents.plugins import getstream, qwen

load_dotenv()


async def create_agent(**kwargs) -> Agent:
    llm = qwen.Realtime(fps=1)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Qwen Assistant", id="agent"),
        instructions="You are a helpful AI assistant. Be friendly and conversational.",
        llm=llm,
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.create_user()
    call = await agent.create_call(call_type, call_id)

    @agent.events.subscribe
    async def on_participant_joined(event: CallSessionParticipantJoinedEvent):
        if event.participant.user.id != "agent":
            await asyncio.sleep(5)
            await agent.simple_response(
                "Tell me a joke, make it extra funny and sarcastic"
            )

    with await agent.join(call):
        await agent.edge.open_demo(call)
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
