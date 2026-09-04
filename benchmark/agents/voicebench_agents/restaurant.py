"""Restaurant tools and agent factory."""

from vision_agents.core import Agent, User
from vision_agents.plugins import getstream

from voicebench_agents import pack_prompt
from voicebench_agents.pipeline import build_llm
from voicebench_agents.world_client import WorldClient


async def create_agent(**kwargs) -> Agent:
    world = WorldClient()
    llm = kwargs.get("llm") or build_llm()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(id="restaurant-agent", name="Copper Spoon Host"),
        instructions=pack_prompt("restaurant"),
        llm=llm,
    )

    @llm.register_function(
        description="Check table availability. time is h:mm 12-hour, like 7:30"
    )
    async def check_availability(
        time: str, party_size: int, patio: bool = False
    ) -> dict:
        return world.call(
            "check_availability", time=time, party_size=party_size, patio=patio
        )

    @llm.register_function(
        description="Book the table. Call this before telling the caller they are booked. allergen is required. time is h:mm 12-hour, like 7:30"
    )
    async def create_reservation(
        time: str,
        party_size: int,
        name: str,
        allergen: str,
        patio: bool = False,
        high_chair: bool = False,
        phone: str = "",
        notes: str = "",
    ) -> dict:
        return world.call(
            "create_reservation",
            time=time,
            party_size=party_size,
            name=name,
            allergen=allergen,
            patio=patio,
            high_chair=high_chair,
            phone=phone,
            notes=notes,
        )

    @llm.register_function(
        description="Update an existing reservation. time is h:mm 12-hour, like 7:30"
    )
    async def update_reservation(
        time: str = "",
        party_size: int = 0,
        allergen: str = "",
        name: str = "",
        phone: str = "",
        patio: bool = False,
        high_chair: bool = False,
    ) -> dict:
        args: dict[str, object] = {}
        if time:
            args["time"] = time
        if party_size:
            args["party_size"] = party_size
        if allergen:
            args["allergen"] = allergen
        if name:
            args["name"] = name
        if phone:
            args["phone"] = phone
        args["patio"] = patio
        args["high_chair"] = high_chair
        return world.call("update_reservation", **args)

    @llm.register_function(description="Look up the menu, including 86'd items")
    async def lookup_menu() -> dict:
        return world.call("lookup_menu")

    @llm.register_function(description="Place a pickup order. allergen is required.")
    async def create_order(
        name: str,
        allergen: str,
        items: list[dict[str, str]],
        pickup_window: str = "",
        modifiers: list[str] | None = None,
    ) -> dict:
        return world.call(
            "create_order",
            name=name,
            allergen=allergen,
            items=items,
            pickup_window=pickup_window,
            modifiers=modifiers or [],
        )

    return agent
