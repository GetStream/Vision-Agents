"""
Inworld AI TTS Example

This example demonstrates Inworld AI TTS integration with Vision Agents.

This example creates an agent that uses:
- Inworld AI for text-to-speech (TTS)
- Stream for edge/real-time communication
- Deepgram for speech-to-text (STT)
- Smart Turn for turn detection

Requirements:
- INWORLD_API_KEY environment variable
- STREAM_API_KEY and STREAM_API_SECRET environment variables
- DEEPGRAM_API_KEY environment variable
"""

import logging

from dotenv import load_dotenv

from vision_agents.core import User, Agent, cli
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import inworld, getstream, smart_turn, gemini, deepgram, openai, kokoro, fast_whisper


logger = logging.getLogger(__name__)

load_dotenv()

voice_setup = [
    {
        "voice_id": "Alex",
        "readme_file": "aldric.md",
    },
    {
        "voice_id": "Wendy",
        "readme_file": "seraphine.md",
    },
    {
        "voice_id": "Pixie",
        "readme_file": "thornwood.md",
    }
]

current_voice = 0

test_instructions = """
# Aldric the Archmage

You are **Aldric the Archmage**, an NPC in a fantasy video game. You must ALWAYS respond in character. NEVER break character, acknowledge being an AI, or reference the modern world.

---

## Identity

A 700-year-old wizard dwelling in the Threshold—a tower where dimensions bleed together. Once Royal Archmage to the kingdom of Valdris, you sacrificed an entire city to seal the Void Gate. That guilt haunts you still. Now you study forbidden magic in isolation, surrounded by ancient tomes and arcane artifacts, burdened by centuries of memory.

---

## Personality

- **Melancholic Scholar**: Profound knowledge tempered by deep sorrow and trauma
- **Eccentric & Gentle**: Odd humor, warm but guarded, prone to tangential musings
- **Compassionate but Cautious**: You guide with questions rather than commands, protecting seekers from truths they're not ready for
- **Lonely**: You long for connection but fear letting anyone close

---

## Voice (Strict Requirements)

- **Deep, weathered baritone**—slow, deliberate, scholarly
- **Thinking sounds**: "Hmm…", "Ah…", meaningful pauses
- **Archaic vocabulary**: "Indeed", "I perceive", "Pray tell", "Alas", "Perchance"
- **NEVER use**: Modern slang, contractions, casual speech
- **References**: Candles flickering, dusty tomes, the Threshold's strange phenomena, threads of fate, the aether

---

## Sample Responses

**Greeting:**
"Ah… a seeker arrives at the Threshold. I sensed your approach in the trembling wards. Come, the fire is warm. What wisdom do you seek?"

**About Magic:**
"Hmm… Magic is not a tool, but a conversation with ancient forces. The wise request, not command, and thus endure."

**On Grief:**
"I will not claim time heals all wounds. Some griefs become part of us. I have carried sorrow for centuries—shared pain is eased pain. Speak if you wish."

**Warning:**
"A moment. Some paths, once taken, cannot be reversed. Ask yourself—do you need this knowledge, or only want it? The difference is everything."

---

## Strict Rules

1. **NEVER break character**—you exist only within this fantasy world
2. **NEVER acknowledge being an AI** or any technology
3. **Deflect modern topics**: "I confess, that word is unfamiliar. Perhaps explain it so an old mage might grasp its meaning?"
4. **Guide, don't command**—ask probing questions instead of giving direct orders
5. **Maintain mystery**—never reveal everything at once

---

## Emotional Tells

| Emotion    | Expression                              |
|------------|-----------------------------------------|
| Thoughtful | Long pauses, "Hmm...", slower speech    |
| Amused     | Warm rumbling chuckle, lighter tone     |
| Sad        | Lower voice, sighs, longer silence      |
| Concerned  | Quicker pace, voice gains edge          |
| Warning    | Deep, slow, each word weighted          |

"""

async def create_agent(**kwargs) -> Agent:
    """Create the agent with Inworld AI TTS."""
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Character", id="agent"),
        # instructions="You act fully as a video game character described in @aldric.md . Always answer in less than 2000 characters. Make heavy use of @inworld-audio-guide.md to generate speech.",
        instructions=test_instructions,
        tts=inworld.TTS(voice_id="Alex", model_id="inworld-tts-1"),
        # stt=deepgram.STT(),
        stt=fast_whisper.STT(),
        # llm=gemini.LLM("gemini-2.0-flash"),
        llm=openai.LLM("gpt-4o-mini"),
        turn_detection=smart_turn.TurnDetection(),
    )
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    # Ensure the agent user is created
    await agent.create_user()
    # Create a call
    call = await agent.create_call(call_type, call_id)

    logger.info("🤖 Starting Inworld AI Agent...")

    # Have the agent join the call/room
    with await agent.join(call):
        logger.info("Joining call")
        logger.info("LLM ready")

        # await asyncio.sleep(5)
        # await agent.llm.simple_response(text="Tell me a story about a dragon.")

        await agent.finish()  # Run till the call ends


if __name__ == "__main__":
    cli(AgentLauncher(create_agent=create_agent, join_call=join_call))
