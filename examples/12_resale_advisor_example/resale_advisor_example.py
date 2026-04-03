"""
Gemma 4 Resale Price Advisor

Point your camera at household items and get resale price suggestions.
Uses Gemma 4 VLM to identify items and search the web for pricing.

Creates an agent that uses:
- TransformersVLM for visual analysis and price estimation (Gemma 4 E4B)
- Deepgram for speech-to-text and text-to-speech
- GetStream for edge/real-time communication

Requirements:
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
- GPU with ~8GB VRAM recommended (or Apple Silicon with 16GB+ unified memory)

First run will download Gemma 4 E4B (~8GB).
"""

import asyncio
import logging
import urllib.parse

import aiohttp
from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, getstream, huggingface

logger = logging.getLogger(__name__)

load_dotenv()

SYSTEM_PROMPT = (
    "You are a resale price advisor. The user is moving house and selling items. "
    "You can see their camera feed. When they ask about an item, identify what you "
    "see and use the search_web tool to look up its current resale value. "
    "Then give a specific price range in USD with a brief justification. "
    "Speak naturally, no lists or formatting. Never use emojis or special characters. "
    "Keep responses under 50 words."
)


async def create_agent(**kwargs) -> Agent:
    """Create an agent that identifies items and searches for resale prices."""
    vlm = huggingface.TransformersVLM(
        model="google/gemma-3n-E4B-it",
        max_new_tokens=200,
        fps=1,
        frame_buffer_seconds=5,
        max_frames=1,
        do_sample=False,
    )

    @vlm.register_function(
        "search_web",
        description="Search the web for resale prices of an item. Pass a search query.",
    )
    async def search_web(query: str) -> str:
        logger.info(f"  [tool] search_web({query!r})")
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (compatible; ResaleAdvisor/1.0)"
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    text = await resp.text()
            snippets: list[str] = []
            for chunk in text.split('class="result__snippet"'):
                if len(snippets) >= 5:
                    break
                if snippets or chunk != text.split('class="result__snippet"')[0]:
                    clean = chunk.split("</a>")[0].split(">")[-1]
                    clean = clean.replace("<b>", "").replace("</b>", "").strip()
                    if clean:
                        snippets.append(clean)
            return "\n".join(snippets) if snippets else f"No results found for: {query}"
        except (aiohttp.ClientError, TimeoutError) as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {e}"

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Resale Price Advisor", id="agent"),
        instructions=SYSTEM_PROMPT,
        llm=vlm,
        tts=deepgram.TTS(),
        stt=deepgram.STT(),
    )

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and run the agent."""
    call = await agent.create_call(call_type, call_id)

    logger.info("Starting Resale Price Advisor...")

    async with agent.join(call):
        await asyncio.sleep(2)
        await agent.llm.simple_response(
            text="Greet the user briefly. Tell them to show items to the camera and ask you for a price.",
        )
        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
