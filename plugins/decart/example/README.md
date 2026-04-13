# Decart Virtual Try-On Example

This example shows you how to build a real-time virtual try-on ("costume") agent using
[Vision Agents](https://visionagents.ai/) and [Decart](https://decart.ai/). The agent listens for voice requests like
"put me in a superhero costume" and uses the Lucy real-time model to restyle the user's video so they appear to be
wearing it, using a reference image.

In this example, the AI wardrobe assistant will:

- Listen to your voice input
- Use [Decart](https://decart.ai/) Lucy to restyle your video feed in real-time with both a prompt and a reference image
- Atomically swap costumes via `processor.update_state(prompt=..., image=...)`
- Fall back to prompt-only outfit changes for freeform requests
- Speak with an expressive voice using [ElevenLabs](https://elevenlabs.io/)
- Run on Stream's low-latency edge network

## Prerequisites

- Python 3.10 or higher
- API keys for:
    - [OpenAI](https://openai.com) (for the LLM)
    - [Decart](https://decart.ai/) (for video restyling)
    - [ElevenLabs](https://elevenlabs.io/) (for text-to-speech)
    - [Deepgram](https://deepgram.com/) (for speech-to-text)
    - [Stream](https://getstream.io/?utm_source=github.com&utm_medium=referral&utm_campaign=vision_agents) (for video/audio infrastructure)

## Installation

1. Install dependencies using uv:
   ```bash
   uv sync
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key
   DECART_API_KEY=your_decart_key
   ELEVENLABS_API_KEY=your_11labs_key
   DEEPGRAM_API_KEY=your_deepgram_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

## Running the Example

Run the agent:

```bash
uv run decart_example.py run
```

The agent will:

1. Create a video call
2. Open a demo UI in your browser
3. Join the call
4. Listen for costume requests and restyle your video with Lucy

## Code Walkthrough

### Setting Up the Agent

The code creates an agent with the Decart processor (Lucy real-time) and a pre-defined set of costumes:

```python
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

processor = decart.RestylingProcessor(
    model="lucy_2_rt",
)
llm = openai.LLM(model="gpt-5")

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Virtual Wardrobe", id="agent"),
    instructions="You are a playful virtual wardrobe assistant...",
    llm=llm,
    tts=elevenlabs.TTS(voice_id="N2lVS1w4EtoT3dr4eOWO"),
    stt=deepgram.STT(),
    processors=[processor],
)
```

**Components:**

- `processor`: The Decart `RestylingProcessor` running the `lucy_2_rt` real-time model, which accepts a reference image.
- `llm`: GPT-5 — picks the right costume and narrates the change.
- `tts` / `stt`: ElevenLabs + Deepgram for a voice-driven loop.

### Swapping Costumes Atomically

`update_state` mirrors the JS SDK's `realtimeClient.set({ prompt, enhance, image })` — prompt and reference image are
applied in a single atomic update so the output video never shows a half-updated state:

```python
@llm.register_function(
    description="Put the user in one of the pre-defined costumes."
)
async def change_costume(name: str) -> str:
    costume = COSTUMES.get(name.lower())
    if costume is None:
        return f"Unknown costume '{name}'. Available: {', '.join(COSTUMES)}."
    await processor.update_state(prompt=costume["prompt"], image=costume["image"])
    return f"Costume changed to {name}."
```

For freeform requests (anything not in `COSTUMES`), the agent calls `change_outfit` which uses
`update_state(prompt=..., image=...)` if the user supplies a URL, or `update_prompt(...)` for prompt-only changes:

```python
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
```

## Customization

### Add or Change Costumes

Edit the `COSTUMES` dict. Each entry needs a `prompt` and an optional `image` — bytes, a file path, an http(s) URL, a
data URI, or a raw base64 string are all accepted.

### Start With a Costume Already On

Pass `initial_image` to the processor so the very first frame is already restyled. Point it at your own hosted image (or
a local file path / bytes / data URI):

```python
processor = decart.RestylingProcessor(
    model="lucy_2_rt",
    initial_prompt="A person wearing a superhero costume",
    initial_image="./costumes/superhero.png",  # or bytes, an http(s) URL, a data URI, or raw base64
)
```

### Change the Voice

Update the `voice_id` in `elevenlabs.TTS` to use a different ElevenLabs voice.

## Learn More

- [Vision Agents Documentation](https://visionagents.ai)
- [Decart Documentation](https://docs.decart.ai)
