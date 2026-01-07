# This is a basic example using Qwen Realtime with Vision Agents
# To run this example, you must have DASHSCOPE_API_KEY set in your env.
# Do note that the model is hosted in Singapore so depending on your location, the latency may vary.
# This model also does not support text input so once you join the call, simply start speaking to the agent.

from dotenv import load_dotenv
from vision_agents.core import Agent, User, cli
from vision_agents.core.agents import AgentLauncher
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

    async with agent.join(call):
        await agent.edge.open_demo(call)
        await agent.finish()


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
