"""Run a vertical agent on Stream WebRTC via Runner.serve()."""

from collections.abc import Awaitable, Callable

from vision_agents.core import Agent, AgentLauncher, Runner

CreateAgent = Callable[..., Awaitable[Agent]]


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        await agent.simple_response(text="Greet the caller briefly and wait.")
        await agent.finish()


def serve_webrtc(
    create_agent: CreateAgent, host: str = "127.0.0.1", port: int = 8000
) -> None:
    Runner(
        AgentLauncher(create_agent=create_agent, join_call=join_call),
    ).serve(host=host, port=port)
