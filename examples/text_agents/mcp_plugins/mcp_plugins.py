import asyncio
import logging
import os
import sys
import webbrowser

from dotenv import load_dotenv
from vision_agents.core import Agent
from vision_agents.plugins import stream

logging.basicConfig(level=logging.INFO)
# What this example prints is the point of it, and a request per HTTP call would bury that.
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A text agent on the router's connectors, plus Blender in the router's sandbox.
TableJourney, which knows where to eat, is a public MCP server outside the catalog, so it is a
custom connector the app's backend creates once.

`agent.yaml` is where all of it is set up, and each kind is set up differently. Sentry,
GitHub, HubSpot and Salesforce are the company's accounts: each entry sets
`connection: company`, the label of the connection created once on the dashboard, and every
conversation uses that account. Linear, Google Calendar, Drive and Docs, Calendly, Cal.com
and Slack are each person's. A name alone is a session binding offering every tool; `tools`
on an entry is the patterns it offers. TableJourney's entry names the custom connector and
the label of its connection. In a chat a login request is a `connector_authorization`
attachment. Blender is no MCP server at all: `sandbox_options` builds it into the router's
Daytona sandbox, and `skills/render.md` is what the subagent writes its scene
under. The render comes back as a file of run_code, attached to the reply.

A calendar belongs to somebody, so the conversation is opened for an end user. Without one
the agent is offered no calendar at all.

The same agent answers on WhatsApp, on a text message and on iMessage, and there is no code
for that here: `channels` in `agent.yaml` names the numbers, the app connected each line once
on the router, and the router answers what arrives. The README has the three steps.

Needs a router: see acceleration/README.md, then point STREAM_ACCELERATION_URL at it. Sentry,
Linear, Calendly and Cal.com register their own clients; GitHub, HubSpot, Slack and Google
need an OAuth client, the app's or the operator's, which the README lists. It also needs
DAYTONA_API_KEY for Blender and Stream credentials to keep the conversation in Chat.
"""

USER_ID = os.environ.get("MCP_PLUGINS_USER_ID", "on-call-engineer")

QUESTION = """
What are my open Linear issues?
"""


async def main(question: str) -> None:
    llm = stream.Accelerated(config="mcp_plugins", user_id=USER_ID)
    agent = Agent(config="mcp_plugins", llm=llm)

    async with agent.chat():
        dashboard = os.environ.get("DASHBOARD_BASE_URL")
        if dashboard:
            url = f"{dashboard.rstrip('/')}/sessions/{llm.router_session_id}"
            print(f"[watch it on the dashboard: {url}]", flush=True)
            webbrowser.open(url, 2)

        await ask_here(agent, question)


async def ask_here(agent: Agent, question: str) -> None:
    """Ask in the terminal and print the answer."""
    async for event in agent.ask(question):
        if event.type == "agent_speech_delta":
            print(event.text, end="", flush=True)
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


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or QUESTION.strip()))
