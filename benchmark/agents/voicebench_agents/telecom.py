"""Telecom tools and agent factory."""

from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, openai

from voicebench_agents import pack_prompt
from voicebench_agents.world_client import WorldClient


async def create_agent(**kwargs) -> Agent:
    world = WorldClient()
    llm = openai.Realtime()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(id="telecom-agent", name="Internet Support"),
        instructions=pack_prompt("telecom"),
        llm=llm,
    )

    @llm.register_function(
        description="Verify the account with PIN, last four, and address"
    )
    async def verify_account(pin: str, last4: str = "", address: str = "") -> dict:
        return world.call("verify_account", pin=pin, last4=last4, address=address)

    @llm.register_function(description="Check for an area outage")
    async def check_outage() -> dict:
        return world.call("check_outage")

    @llm.register_function(description="Walk the customer through a gateway reboot")
    async def walk_reboot() -> dict:
        return world.call("walk_reboot")

    @llm.register_function(description="Open a support ticket")
    async def create_ticket(reason: str, address: str = "") -> dict:
        return world.call("create_ticket", reason=reason, address=address)

    @llm.register_function(
        description="Dispatch a technician. Reboot must have failed first. window is am or pm"
    )
    async def dispatch_tech(window: str, ticket_id: str = "") -> dict:
        return world.call("dispatch_tech", window=window, ticket_id=ticket_id)

    @llm.register_function(description="Apply a bill credit if eligible")
    async def apply_credit(amount: float) -> dict:
        return world.call("apply_credit", amount=amount)

    @llm.register_function(description="Change the service plan after identity")
    async def change_plan(plan: str) -> dict:
        return world.call("change_plan", plan=plan)

    @llm.register_function(description="Store a 3-line warm transfer summary")
    async def create_transfer_summary(summary: str) -> dict:
        return world.call("create_transfer_summary", summary=summary)

    return agent
