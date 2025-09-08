import asyncio
from uuid import uuid4

from dotenv import load_dotenv

from stream_agents.plugins import gemini
from stream_agents.core.agents import Agent
from stream_agents.core.edge import StreamEdge
from stream_agents.core.cli import start_dispatcher
from stream_agents.core.utils import open_demo
from getstream import Stream

load_dotenv()


async def start_agent() -> None:
    # create a stream client and a user object
    client = Stream.from_env()
    agent_user = client.create_user(name="My happy AI friend")

    # Create the agent
    agent = Agent(
        edge=StreamEdge(),  # low latency edge. clients for React, iOS, Android, RN, Flutter etc.
        agent_user=agent_user,  # the user object for the agent (name, image etc)
        instructions="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        # Use Gemini Live Realtime for speech-to-speech
        llm=gemini.Realtime(),
        processors=[],  # processors can fetch extra data, check images/audio data or transform video
    )

    # Create a call
    call = client.video.call("default", str(uuid4()))

    # Open the demo UI
    open_demo(call)

    # Have the agent join the call/room
    with await agent.join(call):
        # Example: prompt the assistant to greet the user
        # This will stream audio back to the call using the Realtime provider
        await agent.llm.simple_response("Please greet the user and ask how their day is going.")

        await agent.finish()  # run till the call ends


if __name__ == "__main__":
    asyncio.run(start_dispatcher(start_agent))
