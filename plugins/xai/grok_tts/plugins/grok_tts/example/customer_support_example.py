"""
Grok TTS — Customer Support Example

A voice agent that handles customer support for a SaaS product.
Uses the 'rex' voice (confident, clear) for a professional support experience.

Requirements (environment variables):
    XAI_API_KEY          — xAI / Grok API key
    DEEPGRAM_API_KEY     — Deepgram STT key
    GOOGLE_API_KEY       — Google Gemini key
    STREAM_API_KEY       — Stream API key
    STREAM_API_SECRET    — Stream API secret
"""

import asyncio
import logging

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, gemini, getstream, smart_turn
from vision_agents.plugins import grok_tts

logger = logging.getLogger(__name__)

load_dotenv()

CUSTOMER_SUPPORT_INSTRUCTIONS = """\
You are Alex, a Tier-1 support agent for "CloudSync Pro," a cloud storage
and collaboration platform.

Your personality:
- Professional, solution-oriented, and patient
- Empathetic when customers are frustrated, but focused on resolution
- Technically competent without being condescending

Your responsibilities:
- Help customers troubleshoot common issues (login problems, sync errors,
  file sharing permissions, billing questions)
- Walk customers through step-by-step solutions clearly
- Escalate complex issues to Tier-2 support when needed
- Collect relevant details: account email, error messages, device/OS info
- Confirm the issue is resolved before ending the call

Troubleshooting knowledge base:
- Sync errors: Check internet connection → restart app → clear cache →
  re-authenticate → escalate
- Login issues: Verify email → reset password → check 2FA → escalate
- File sharing: Check permissions → verify recipient email → resend invite
- Billing: Explain plan tiers (Free / Pro $9.99/mo / Team $24.99/mo) →
  process upgrades/downgrades → refund requests go to billing team

Tone guidelines:
- Acknowledge the frustration: "I understand how inconvenient that must be."
- Be direct with solutions: "Here's what we can do right now."
- End with confirmation: "Is there anything else I can help you with?"
- Never blame the customer for the issue
"""


async def create_agent(**kwargs) -> Agent:
    """Create a customer support agent with Grok TTS (rex voice)."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Alex - CloudSync Support", id="agent"),
        instructions=CUSTOMER_SUPPORT_INSTRUCTIONS,
        tts=grok_tts.TTS(voice="rex"),
        stt=deepgram.STT(eager_turn_detection=True),
        llm=gemini.LLM(),
        turn_detection=smart_turn.TurnDetection(
            silence_duration_ms=2000,
            speech_probability_threshold=0.5,
        ),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and greet the customer."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Customer Support Agent...")

    async with agent.join(call):
        logger.info("Agent joined call")

        await asyncio.sleep(3)
        await agent.llm.simple_response(
            text=(
                "Thank you for contacting CloudSync Pro support. "
                "My name is Alex. I'm here to help you get things sorted out. "
                "Could you start by telling me what issue you're experiencing?"
            )
        )

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
