import asyncio
import logging
import os
import uuid

import click
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from twilio.rest import Client

from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream, twilio

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NGROK_URL = os.environ["NGROK_URL"]

app = FastAPI()
call_registry = twilio.TwilioCallRegistry()


async def create_agent() -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(id="ai-agent", name="AI Assistant"),
        instructions="Speak english. Keep your replies short. You're calling a restaurant and want to make a dinner reservation. for 4, today between 18:00 or 19:00 pm. ideally 18:30",
        llm=gemini.Realtime(),
    )


async def initiate_outbound_call(from_number: str, to_number: str) -> str:
    """Initiate an outbound call via Twilio. Returns the call_id."""
    twilio_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

    call_id = str(uuid.uuid4())

    async def prepare_call():
        agent = await create_agent()
        phone_user = User(name=f"Outbound call {call_id[:8]}", id=f"phone-{call_id}")

        # Create both users in a single API call
        await agent.edge.create_users([agent.agent_user, phone_user])

        stream_call = await agent.create_call("default", call_id)
        agent_session = await agent.join(stream_call, wait_for_participant=False)
        logger.info("prepared the call, ready to start")
        return agent, phone_user, stream_call, agent_session

    twilio_call = call_registry.create(call_id, prepare=prepare_call)
    url = f"wss://{NGROK_URL}/twilio/media/{call_id}/{twilio_call.token}"
    logger.info(f"Forwarding to media url: {url} \n %s", twilio.create_media_stream_twiml(url))

    twilio_client.calls.create(
        twiml=twilio.create_media_stream_twiml(url),
        to=to_number,
        from_=from_number,
    )
    logger.info(f"📞 Initiated call {call_id} from {from_number} to {to_number}")
    return call_id


@app.websocket("/twilio/media/{call_sid}/{token}")
async def media_stream(websocket: WebSocket, call_sid: str, token: str):
    twilio_call = call_registry.validate(call_sid, token)

    logger.info(f"🔗 Media stream connected for call {call_sid}")

    twilio_stream = twilio.TwilioMediaStream(websocket)
    await twilio_stream.accept()
    twilio_call.twilio_stream = twilio_stream

    try:
        agent, phone_user, stream_call, agent_session = await twilio_call.await_prepare()
        twilio_call.stream_call = stream_call

        await twilio.attach_phone_to_call(stream_call, twilio_stream, phone_user.id)

        with agent_session:
            await agent.llm.simple_response(
                text="Ask to reserve and answer any follow up questions as needed"
            )
            await twilio_stream.run()
    finally:
        call_registry.remove(call_sid)


async def run_with_server(from_number: str, to_number: str):
    """Start the server and initiate the outbound call once ready."""
    config = uvicorn.Config(app, host="localhost", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # Start server in background task
    server_task = asyncio.create_task(server.serve())

    # Wait for server to be ready
    while not server.started:
        await asyncio.sleep(0.1)

    logger.info("🚀 Server ready, initiating outbound call...")

    # Initiate the outbound call
    await initiate_outbound_call(from_number, to_number)

    # Keep running until server shuts down
    await server_task


@click.command()
@click.option("--from", "from_number", required=True, help="The phone number to call from. Needs to be active in your Twilio account")
@click.option("--to", "to_number", required=True, help="The phone number to call")
def main(from_number: str, to_number: str):
    logger.info("Starting outbound example. Note that latency is higher in dev. Deploy to US east for low latency")
    asyncio.run(run_with_server(from_number, to_number))


if __name__ == "__main__":
    main()