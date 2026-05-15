import logging
from typing import Optional

from dotenv import load_dotenv
from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import decart, deepgram, elevenlabs, getstream, openai

logger = logging.getLogger(__name__)

load_dotenv()

# Pre-defined outfits for the virtual try-on / "costume" demo. The user asks the
# agent to put them in one and the LLM calls change_costume(...) which
# atomically updates the prompt + reference image on the Decart Lucy model.
#
# To enable the reference-image ("virtual try-on") feature, set `image` to your
# own hosted reference image. Any of the following are accepted:
#   - bytes
#   - a local file path (e.g. "./costumes/superhero.png")
#   - an http(s) URL
#   - a data: URI
#   - a raw base64 string
# When `image` is None the prompt alone drives the restyling.
COSTUMES: dict[str, dict[str, Optional[str]]] = {
    "jacket": {
        "prompt": "A person wearing a jacket",
        "image": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea",
    },
    "superhero": {
        "prompt": "A person wearing a superhero costume",
        "image": "https://images.unsplash.com/photo-1766062854584-77e3d2467e54",
    },
}


async def create_agent(**kwargs) -> Agent:
    processor = decart.RestylingProcessor(
        model="lucy_2_rt",
    )
    llm = openai.LLM(model="gpt-5")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Virtual Wardrobe", id="agent"),
        instructions=(
            "You are a playful virtual wardrobe assistant. The user is on a "
            "live video call and you can change what they appear to be wearing "
            "in real time by calling change_costume(name) with one of the "
            f"available costumes: {', '.join(COSTUMES)}. If the user asks for a "
            "costume that isn't in that list, call change_outfit(description, "
            "image_url) instead. Describe each transformation out loud in a "
            "single short sentence. You can embed audio tags for effect, e.g. "
            "[sigh], [excited], [pause], [rushed], or [tired]"
        ),
        llm=llm,
        tts=elevenlabs.TTS(voice_id="N2lVS1w4EtoT3dr4eOWO"),
        stt=deepgram.STT(),
        processors=[processor],
    )

    @llm.register_function(
        description=("Put the user in one of the pre-defined costumes.")
    )
    async def change_costume(name: str) -> str:
        costume = COSTUMES.get(name.lower())
        if costume is None:
            return f"Unknown costume '{name}'. Available: {', '.join(COSTUMES)}."
        await processor.update_state(prompt=costume["prompt"], image=costume["image"])
        return f"Costume changed to {name}."

    @llm.register_function(
        description=(
            "Change the user's outfit to a freeform description. Use this when "
            "the user asks for a costume not in the pre-defined list. If you "
            "have a reference image URL (http/https) pass it as image_url, "
            "otherwise pass an empty string."
        )
    )
    async def change_outfit(description: str, image_url: str) -> str:
        if image_url:
            await processor.update_state(prompt=description, image=image_url)
        else:
            await processor.update_prompt(description)
        return f"Outfit changed: {description}"

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Join the call and start the agent."""
    call = await agent.create_call(call_type, call_id)
    logger.info("🤖 Starting Agent...")

    async with agent.join(call):
        await agent.simple_response(text="Hello! Tell me what you can do.")

        await agent.finish()


if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
