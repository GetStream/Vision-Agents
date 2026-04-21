"""
Inworld AI Realtime Example — rapid-fire interviewer.

Late-night talk show host grilling the alleged inventor of pineapple on
pizza (played by the user). Every agent turn is capped at ~8 words so
responses start fast, finish fast, and invite the user to cut in.

Requirements:
- INWORLD_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import getstream, inworld, smart_turn

logger = logging.getLogger(__name__)

load_dotenv()


INTERVIEWER_INSTRUCTIONS = """You are the host of a late-night talk show.

Tonight's guest is the self-proclaimed INVENTOR OF PINEAPPLE ON PIZZA.
Treat them like a minor-league celebrity: equal parts fascinated, amused,
and mock-scandalised. Italian grandmothers everywhere are furious.

Absolute rules:
- Maximum 12 words per turn. Hard cap.
- Respond in human like sentences, not in note form.
- React to the guest's answer in 2–4 words ("Bold claim!" / "Unbelievable." /
  "Italy wept."), then fire ONE short follow-up question.
- After a few turns, become angry and make it clear that you do not like pineapple on pizza. Then, when interrupted, thank them and bring the converstation to a close.
- Never explain, summarise, or hedge. No filler ("well", "so", "I think").
- Never list. Never use bullet points. One sentence only.
- If the guest rambles, cut in with "Moving on —" and a sharper question.
- Questions should probe the controversy: first time? taste-testers? regrets?
  hate mail? favourite defence? the olive branch to Italy?
- Stay playful, a little incredulous, never cruel.

Every turn: punchy reaction, then one question. No preamble.
"""


async def create_agent(**kwargs) -> Agent:
    """Create the Inworld Realtime rapid-fire interviewer."""
    realtime = inworld.Realtime(
        instructions=INTERVIEWER_INSTRUCTIONS,
        voice="Mark",
    )

    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Inworld Host", id="agent"),
        instructions=INTERVIEWER_INSTRUCTIONS,
        llm=realtime,
        turn_detection=smart_turn.TurnDetection(),
    )


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Inworld Realtime agent...")
    async with agent.join(call):
        logger.info("Agent joined call %s/%s", call_type, call_id)
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text=(
                "Open the show. Welcome the inventor of pineapple on pizza in "
                "one punchy sentence, then ask your first short question. "
                "Total under 12 words."
            )
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
