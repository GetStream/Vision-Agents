# Decart Storyteller Example

This example shows you how to build a real-time storytelling agent using [Vision Agents](https://visionagents.ai/)
and [Decart](https://decart.ai/). The agent tells a story while transforming your video feed into an animated style that
matches the narrative.

In this example, the AI storyteller will:

- Listen to your voice input
- Generate a story based on your interactions
- Use [Decart](https://decart.ai/) to restyle your video feed in real-time (e.g., "A cute animated movie with vibrant
  colours")
- Change the video style dynamically as the story progresses
- Speak with an expressive voice using [ElevenLabs](https://elevenlabs.io/)
- Run on Stream's low-latency edge network

## Prerequisites

- Python 3.10 or higher
- API keys for:
    - [OpenAI](https://openai.com) (for the LLM)
    - [Decart](https://decart.ai/) (for video restyling)
    - [ElevenLabs](https://elevenlabs.io/) (for text-to-speech)
    - [Deepgram](https://deepgram.com/) (for speech-to-text)
    - [Stream](https://getstream.io/) (for video/audio infrastructure)

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
4. Start telling a story and restyling your video

## Code Walkthrough

### Setting Up the Agent

The code creates an agent with the Decart processor and other components:

```python
processor = decart.RestylingProcessor(
    initial_prompt="A cute animated movie with vibrant colours",
    model="mirage_v2"
)

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="Story teller", id="agent"),
    instructions="You are a story teller...",
    llm=openai.LLM(model="gpt-4o-mini"),
    tts=elevenlabs.TTS(voice_id="N2lVS1w4EtoT3dr4eOWO"),
    stt=deepgram.STT(),
    processors=[processor],
)
```

**Components:**

- `processor`: The Decart RestylingProcessor that transforms the video feed.
- `llm`: The language model (GPT-4o-mini) that generates the story and controls the processor.
- `tts`: ElevenLabs TTS for expressive voice output.
- `stt`: Deepgram STT for transcribing user speech.
- `processors`: The list of video processors (just Decart in this case).

### Dynamic Style Changing

The agent can change the video style dynamically using a registered function:

```python
@llm.register_function(
    description="This function changes the prompt of the Decart processor which in turn changes the style of the video and user's background"
)
async def change_prompt(prompt: str) -> str:
    await processor.update_prompt(prompt)
    return f"Prompt changed to {prompt}"
```

This allows the LLM to call `change_prompt("A dark and stormy night")` to instantly change the visual style of the video
to match the story's mood.

## Customization

### Change the Initial Style

Modify the `initial_prompt` in the `RestylingProcessor` to start with a different look:

```python
processor = decart.RestylingProcessor(
    initial_prompt="A cyberpunk city with neon lights",
    model="mirage_v2"
)
```

### Modify the Storytelling Persona

Edit the `instructions` passed to the `Agent` to change the storyteller's personality, tone, or the type of stories they
tell.

### Change the Voice

Update the `voice_id` in `elevenlabs.TTS` to use a different ElevenLabs voice.

## Learn More

- [Vision Agents Documentation](https://visionagents.ai)
- [Decart Documentation](https://docs.decart.ai)
