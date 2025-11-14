# Voice Output Options for Faster Response

## The Problem

`agent.llm.simple_response()` is slow because:
1. It sends text to Gemini's LLM
2. Gemini generates audio through its speech synthesis
3. This adds significant latency (often 1-3 seconds)

## Solution Options

### Option 1: Use `agent.say()` with TTS Provider (FASTEST) ⚡

**Best for:** Fast, low-latency voice responses

Add a TTS provider to your agent and use `agent.say()`:

```python
from vision_agents.plugins import elevenlabs, cartesia, kokoro

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="AI squat coach"),
    instructions="Read @squat-coach.md",
    llm=gemini.Realtime(fps=LLM_CONFIG["fps"]),
    tts=elevenlabs.TTS(),  # Add TTS provider here
    processors=[...],
)

# In your event handler:
await agent.say(message)  # Fast! Uses TTS directly
```

**Available TTS Providers:**
- **ElevenLabs** (`elevenlabs.TTS()`) - High quality, fast
- **Cartesia** (`cartesia.TTS()`) - Very low latency
- **Kokoro** (`kokoro.TTS()`) - Fast, good quality
- **AWS Polly** (`aws.PollyTTS()`) - AWS-based

**Latency:** ~100-300ms (vs 1-3 seconds with LLM)

### Option 2: Use `agent.say()` without TTS (FALLBACK)

If no TTS is configured, `agent.say()` will:
- Send the message to conversation (chat)
- But won't speak (no TTS available)

You'll see a warning: "No TTS available, cannot synthesize speech"

### Option 3: Use `agent.llm.simple_response()` (SLOWEST)

This is what you're currently using. It's slow because:
- Goes through Gemini's LLM
- LLM generates audio (slower than dedicated TTS)

**Latency:** 1-3 seconds

### Option 4: Use Gemini's Native API Directly

For Gemini Realtime, you can use the native method:

```python
# This is what simple_response does internally
await agent.llm.send_realtime_input(text=message)
```

**Note:** This has the same latency as `simple_response()` since it's the same underlying call.

## Recommended Setup

For your squat counter, add a TTS provider:

```python
from vision_agents.plugins import getstream, gemini, elevenlabs  # or cartesia, kokoro

agent = Agent(
    edge=getstream.Edge(),
    agent_user=User(name="AI squat coach"),
    instructions="Read @squat-coach.md",
    llm=gemini.Realtime(fps=LLM_CONFIG["fps"]),  # For conversation
    tts=elevenlabs.TTS(),  # For fast voice responses
    processors=[...],
)
```

Then in your event handler:
```python
await agent.say(message)  # Fast TTS voice output
```

## Why This Works

- **TTS providers** are optimized for fast text-to-speech conversion
- **They don't go through the LLM** - direct text → audio
- **They can stream audio chunks** immediately
- **Lower latency** - typically 100-300ms vs 1-3 seconds

## Cost Considerations

- **TTS providers** charge per character/request (usually very cheap)
- **Gemini Realtime** charges for LLM usage (more expensive)
- Using TTS for simple responses saves money and is faster

## Current Code Status

The code now uses `agent.say()` which will:
- Use TTS if you add a TTS provider (fast)
- Fall back gracefully if no TTS is configured (no voice, but chat still works)

To enable fast voice output, uncomment one of the TTS imports and add it to the Agent initialization.

