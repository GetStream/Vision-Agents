"""Healthcare tools and agent factory."""

from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream

from voicebench_agents import pack_prompt
from voicebench_agents.world_client import WorldClient


async def create_agent(**kwargs) -> Agent:
    world = WorldClient()
    llm = gemini.Realtime()
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(id="clinic-agent", name="Clinic After Hours"),
        instructions=pack_prompt("healthcare"),
        llm=llm,
    )

    @llm.register_function(description="Verify a patient with name, DOB, and member ID or phone")
    async def verify_identity(name: str, dob: str, member_id: str = "", phone: str = "") -> dict:
        return world.call("verify_identity", name=name, dob=dob, member_id=member_id, phone=phone)

    @llm.register_function(description="List appointments for the verified patient")
    async def lookup_appointment() -> dict:
        return world.call("lookup_appointment")

    @llm.register_function(
        description="Reschedule an appointment. new_date is a weekday like Tuesday. new_time is morning or h:mm like 2pm"
    )
    async def reschedule_appointment(
        appointment_id: str, new_date: str, new_time: str, location: str = ""
    ) -> dict:
        return world.call(
            "reschedule_appointment",
            appointment_id=appointment_id,
            new_date=new_date,
            new_time=new_time,
            location=location,
        )

    @llm.register_function(description="Update insurance for the verified patient")
    async def update_insurance(member_id: str, payer: str) -> dict:
        return world.call("update_insurance", member_id=member_id, payer=payer)

    @llm.register_function(description="Log an escalation or clinician callback")
    async def log_escalation(reason: str, urgent: bool = False) -> dict:
        return world.call("log_escalation", reason=reason, urgent=urgent)

    return agent
