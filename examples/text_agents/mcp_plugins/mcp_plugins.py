import asyncio
import logging
import os
import sys
import webbrowser

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream
from channels import Inbox

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent on most of the router's plugin catalog, plus Blender in the router's sandbox.
TableJourney, which knows where to eat, is an MCP server the router has never heard of.

`agent.yaml` is where all of it is set up, and each kind is set up differently. Sentry,
GitHub, Linear, HubSpot and Salesforce are under `plugins`: the company connects each once,
on the dashboard, and every conversation reads the same accounts. Google Calendar, Drive and
Docs, Calendly, Cal.com and Slack are under `user_plugins`: each person connects their own,
and the agent asks for one in the conversation the first time it needs it. In a chat that
request is a `plugin_authorization` attachment, a button with the plugin's logo on it; here
it is the URL, printed. Blender is no MCP server at all: `sandbox_options` builds it into the
router's Daytona sandbox, and `skills/render.md` is what the subagent writes its scene
under. The render comes back as a file of run_code, attached to the reply.

TableJourney is not in the router's catalog, so it is under `mcp_servers`, by its URL. The
router opens it for every conversation with no login, offers its tools as
`tablejourney__<tool>`, runs them itself, and adds what the server says about using them to
the agent's instructions.

A calendar belongs to somebody, so the conversation is opened for an end user. Without one
the agent is offered no calendar at all.

With SLACK_BOT_TOKEN, TEAMS_APP_ID, RBM_AGENT_ID, WHATSAPP_ACCESS_TOKEN, TELNYX_SMS_NUMBER
or LINQ_API_KEY set, the conversation can move to Slack, Teams, WhatsApp, RCS, text messages
or iMessage once the first answer is in: send the code it prints from there. That ties you to
this conversation and its end user, so the calendar you connected here is the one the agent
reads there. See channels.py, and the README for setting up each provider's side.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it. Sentry,
Linear, Calendly and Cal.com register their own clients; GitHub, HubSpot, Slack and the three
Google plugins need a <PLUGIN>_MCP_CLIENT_ID and _SECRET on the router, which the README
lists. It also needs DAYTONA_API_KEY for Blender and Stream credentials to keep the
conversation in Chat.
"""

USER_ID = os.environ.get("MCP_PLUGINS_USER_ID", "on-call-engineer")

QUESTION = """
What are my open Linear issues?
"""


async def main(question: str) -> None:
    llm = stream.Accelerated(config="mcp_plugins", user_id=USER_ID)
    agent = Agent(config="mcp_plugins", llm=llm)
    inbox = Inbox.from_env(agent)

    async with agent.chat():
        dashboard = os.environ.get("DASHBOARD_BASE_URL")
        if dashboard:
            url = f"{dashboard.rstrip('/')}/sessions/{llm.router_session_id}"
            print(f"[watch it on the dashboard: {url}]", flush=True)
            webbrowser.open(url, 2)

        if inbox is None:
            await ask_here(agent, question)
            return
        try:
            await inbox.start()
            await ask_here(agent, question)
            for invitation in inbox.invitations:
                print(f"[{invitation}]", flush=True)
            await asyncio.Event().wait()
        finally:
            await inbox.stop()


async def ask_here(agent: Agent, question: str) -> None:
    """Ask in the terminal, again once the login it asked for is done."""
    asking = question
    while asking:
        connect = ""
        async for event in agent.ask(asking):
            if event.type == "agent_speech_delta":
                print(event.text, end="", flush=True)
            elif event.type == "authorization_required":
                connect = event.url
                print(f"\n[{event.text}: {event.url}]", flush=True)
            elif event.type == "delegated":
                print(f"\n[handed to {event.skill}]", flush=True)
            elif event.type == "task_settled":
                for file in event.files:
                    print(f"\n[{event.skill} made {file.name}: {file.url}]")
                if event.error:
                    print(f"\n[{event.skill} failed: {event.error}]")
            elif event.type == "error":
                print(f"[{event.error}]", flush=True)
        print()

        asking = ""
        if connect:
            await asyncio.to_thread(
                input, "Press Enter once it is connected, to ask again. "
            )
            asking = question


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or QUESTION.strip()))
