import asyncio
import logging
import os
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from vision_agents.plugins import stream

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

"""
A travel support chat agent, set up from this directory and tested on the dashboard.

`instructions.md` is who it is, `knowledge/travel_policy.md` what it knows about bookings,
refunds and baggage, and `skills/itinerary.md` how it plans a trip. TableJourney, a public
MCP server with real restaurants, sights and stays, is a custom connector the app creates
once; `agent.yaml` binds it and names the tools the agent may call.

`lookup_booking` is a function of this process: it stays running and hosts it, so every
conversation with the agent can call it, the ones the dashboard opens included.

This script makes sure the connector exists, syncs the agent and opens the dashboard's test
page, where you chat with it. Needs a router (STREAM_ACCELERATION_URL), Stream credentials
and DASHBOARD_BASE_URL, the app's agents page on the dashboard.
"""

HERE = Path(__file__).parent
BOOKINGS = {
    "WF7K2Q": {
        "traveller": "Ana Silva",
        "trip": "Amsterdam to Lisbon, 12 to 15 June",
        "flight": "TP 667, departs 07:05",
        "hotel": "Casa do Bairro, 3 nights",
        "checked_bags": 1,
        "insurance": False,
        "status": "confirmed",
    },
}
CONNECTOR = {
    "id": "custom_tablejourney",
    "name": "TableJourney",
    "description": "Verified restaurants, food festivals and food tours",
    "endpoint": "https://tablejourney.com/mcp",
    "schemes": ["none"],
}
CONNECTION_LABEL = "tablejourney"

agent = stream.Client().agent("travel_support")


@agent.register()
async def lookup_booking(reference: str) -> dict[str, object]:
    """Look up a booking by its six-character booking reference, such as WF7K2Q."""
    booking = BOOKINGS.get(reference.strip().upper())
    if booking is None:
        return {"error": f"no booking with reference {reference}"}
    return {"reference": reference.strip().upper(), **booking}


async def main() -> None:
    await ensure_tablejourney(stream.Backend())
    synced = await stream.sync_agent("travel_support", path=str(HERE))

    url = f"{os.environ['DASHBOARD_BASE_URL'].rstrip('/')}/{synced.config.id}/test?channel=text"
    print(f"Chat with travel_support: {url}")
    webbrowser.open(url, 2)

    dispatch = stream.Dispatch()
    dispatch.host(agent)
    print("Hosting lookup_booking; ctrl-c to stop.")
    await dispatch.run()


async def ensure_tablejourney(backend: stream.Backend) -> None:
    """Create the TableJourney connector and the app's connection to it, once."""
    async with backend.client().get_async_httpx_client() as http:
        # Creating it again with the same definition stores nothing new.
        await http.post("/v1/agents/connectors", json=CONNECTOR)

        listed = await http.get(
            "/v1/agents/connections",
            params={"connector_id": CONNECTOR["id"], "owner_type": "app"},
        )
        if any(c.get("label") == CONNECTION_LABEL for c in listed.json()["items"]):
            return

        created = await http.post(
            "/v1/agents/connections",
            json={
                "connector_id": CONNECTOR["id"],
                "owner": {"type": "app"},
                "label": CONNECTION_LABEL,
            },
        )
        connection = created.json()
        # A connector that takes no login is connected by storing empty credentials.
        await http.put(
            f"/v1/agents/connections/{connection['id']}/credentials",
            json={"expected_revision": connection["revision"]},
        )


if __name__ == "__main__":
    asyncio.run(main())
