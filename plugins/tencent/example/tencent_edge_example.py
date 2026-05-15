import os
from urllib.parse import urlencode

import TLSSigAPIv2
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import cartesia, elevenlabs, gemini, tencent

load_dotenv()

TEST_CLIENT_USER_ID = "test-user-1"
TEST_CLIENT_BASE_URL = os.environ.get(
    "TENCENT_TEST_CLIENT_URL", "http://localhost:8000/test_client.html"
)


async def create_agent(**kwargs) -> Agent:
    sdk_app_id = int(os.environ["TENCENT_SDKAppID"])
    secret_key = os.environ["TENCENT_SDKSecretKey"]

    agent = Agent(
        edge=tencent.Edge(sdk_app_id=sdk_app_id, key=secret_key),
        agent_user=User(name="Tencent Voice Agent", id="tencent-voice-agent"),
        instructions="You are a helpful voice assistant. Respond concisely.",
        llm=gemini.LLM(model="gemini-3.1-flash-lite-preview"),
        tts=cartesia.TTS(),
        stt=elevenlabs.STT(),
    )
    return agent


def _build_test_client_url(sdk_app_id: int, secret_key: str, room_id: str) -> str:
    sig = TLSSigAPIv2.TLSSigAPIv2(sdk_app_id, secret_key).gen_sig(TEST_CLIENT_USER_ID)
    params = urlencode(
        {
            "appid": sdk_app_id,
            "user": TEST_CLIENT_USER_ID,
            "room": room_id,
            "sig": sig,
        }
    )
    return f"{TEST_CLIENT_BASE_URL}?{params}"


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.authenticate()
    # Use a fixed room ID so the browser test client can join it ahead of the
    # agent. ElevenLabs closes its STT websocket after ~15s of silence, so the
    # browser needs to be in the room before the agent connects.
    room_id = os.environ.get("TENCENT_TEST_ROOM_ID", "12345")
    sdk_app_id = int(os.environ["TENCENT_SDKAppID"])
    secret_key = os.environ["TENCENT_SDKSecretKey"]
    test_url = _build_test_client_url(sdk_app_id, secret_key, room_id)
    print(f"\n🔗 Open the test client and click Join:\n    {test_url}\n", flush=True)

    call = await agent.create_call(call_type, room_id)

    async with agent.join(call, participant_wait_timeout=None):
        await agent.simple_response(
            "Hi! I'm the Tencent voice agent. Say something to test the connection."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
