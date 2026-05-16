import os

import TLSSigAPIv2
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import cartesia, elevenlabs, gemini, tencent

load_dotenv()

TEST_CLIENT_USER_ID = "test-user-1"
TENCENT_QUICK_DEMO_URL = (
    "https://web.sdk.qcloud.com/trtc/webrtc/v5/demo/quick-demo-js/index.html"
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


def _print_test_client_instructions(
    sdk_app_id: int, secret_key: str, room_id: str
) -> None:
    sig = TLSSigAPIv2.TLSSigAPIv2(sdk_app_id, secret_key).gen_sig(TEST_CLIENT_USER_ID)
    print(
        "\n🔗 Open the Tencent TRTC quick demo and paste these values, then click Enter Room:\n"
        f"    URL:     {TENCENT_QUICK_DEMO_URL}\n"
        f"    AppID:   {sdk_app_id}\n"
        f"    UserID:  {TEST_CLIENT_USER_ID}\n"
        f"    RoomID:  {room_id}\n"
        f"    UserSig: {sig}\n",
        flush=True,
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await agent.authenticate()
    # Use a fixed room ID so the browser test client can join it ahead of the
    # agent. ElevenLabs closes its STT websocket after ~15s of silence, so the
    # browser needs to be in the room before the agent connects.
    room_id = os.environ.get("TENCENT_TEST_ROOM_ID", "12345")
    sdk_app_id = int(os.environ["TENCENT_SDKAppID"])
    secret_key = os.environ["TENCENT_SDKSecretKey"]
    _print_test_client_instructions(sdk_app_id, secret_key, room_id)

    call = await agent.create_call(call_type, room_id)

    async with agent.join(call, participant_wait_timeout=None):
        await agent.simple_response(
            "Hi! I'm the Tencent voice agent. Say something to test the connection."
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
