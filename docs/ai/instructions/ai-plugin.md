
# Plugin Development Guide

## Folder Structure

Every plugin should follow this structure, an example for the plugin named elevenlabs:

```
/plugins/elevenlabs
- pyproject.toml
- README.md
- py.typed
- stream_agents/plugins/
  - elevenlabs/
    - __init__.py
    - tts.py
    - events.py
    - tests/
```

And the logic for the plugin should live in `/plugins/elevenlabs/stream_agents/plugins/...`

## Naming

When a plugin is imported it's used like:

```python
from stream_agents.plugins import elevenlabs, anthropic

tts = elevenlabs.TTS()
llm = anthropic.LLM()
```

## Event System Overview

Plugins use events to communicate asynchronously with other components. Events flow through the system to enable real-time communication and monitoring.

### What Events Do

Events serve as the communication backbone of stream-agents:

- **STT Events**: Speech-to-text results, partial transcripts, errors
- **TTS Events**: Audio generation, synthesis status, completion
- **LLM Events**: Text responses, streaming deltas, function calls
- **VAD Events**: Voice activity detection, speech start/end
- **Agent Events**: Agent speech requests, conversation tracking
- **Plugin Events**: Custom plugin-specific events

### Event Flow in Plugins

```python
# STT Plugin Flow
Audio Input → STT Plugin → STTTranscriptEvent → LLM Plugin
                                    ↓
                              LLMResponseEvent → TTS Plugin
                                    ↓
                              TTSAudioEvent → Audio Output

# Agent Flow  
User Input → Agent.say() → AgentSayEvent → TTS Plugin → Audio Output
```

### Where Events Are Sent

**From Plugins:**
- `self.events.send(event)` - Send events from your plugin
- Events are automatically forwarded to the agent's event manager

**To Other Components:**
- **Agent**: Receives all events from all plugins
- **Other Plugins**: Can subscribe to events from any plugin
- **External Systems**: Can subscribe to events for monitoring/logging

### Plugin Event Integration

**Simple Plugins** (use base events):
```python
class MySTT(STT):
    async def transcribe(self, audio):
        result = await self._call_api(audio)
        
        # Send base STT event
        self.events.send(STTTranscriptEvent(
            plugin_name="mystt",
            text=result.text,
            confidence=result.confidence
        ))
```

**Complex Plugins** (custom events):
```python
class MyLLM(LLM):
    def __init__(self):
        super().__init__()
        # Register custom events
        self.events.register_events_from_module(events)
    
    async def generate_response(self, text):
        # Send custom events
        self.events.send(MyLLMProcessingEvent(
            plugin_name="myllm",
            status="started"
        ))
        
        result = await self._call_api(text)
        
        self.events.send(MyLLMResponseEvent(
            plugin_name="myllm",
            response=result
        ))
```

### Event Subscription Patterns

**Subscribe to Events:**
```python
@agent.events.subscribe
async def handle_transcript(event: STTTranscriptEvent):
    print(f"User said: {event.text}")

@agent.events.subscribe  
async def handle_llm_response(event: LLMResponseEvent):
    print(f"LLM responded: {event.text}")
```

**Cross-Plugin Communication:**
```python
# Plugin A can listen to Plugin B's events
@self.events.subscribe
async def handle_other_plugin_event(event: OtherPluginEvent):
    # React to events from other plugins
    pass
```

### Event Types Available

**Core Events:**
- `STTTranscriptEvent` - Speech transcription results
- `TTSAudioEvent` - Generated audio data
- `LLMResponseEvent` - Language model responses
- `VADSpeechStartEvent` / `VADSpeechEndEvent` - Voice activity
- `AgentSayEvent` - Agent speech requests

**Plugin Events:**
- `OpenAIStreamEvent` - OpenAI streaming events
- `GeminiConnectedEvent` - Gemini connection status
- `DeepgramTranscriptEvent` - Deepgram transcription
- Custom events defined by your plugin

For detailed event system implementation, see [ai-events-example.md](ai-events-example.md).

## Example Plugin

An example plugin is located in `plugins/example`. Copying the example is the best way to create a new plugin. After copying the example be sure to:

- Update the folder name "example" to your plugin's name
- Open `pyproject.toml` and update the name, description etc
- Update the event types in your `events.py` file
- Register your events in the plugin's `__init__` method

## Guidelines

When building the plugin read these guides:

- **TTS**: [ai-tts.md](ai-tts.md)
- **STT**: [ai-stt.md](ai-stt.md)  
- **STS/realtime/LLM**: [ai-llm.md](ai-llm.md)
- **Video processor**: [ai-video-processor.md](ai-video-processor.md)

## Update pyproject.toml

Be sure to update `pyproject.toml` at the root of this project. Add the new plugin to:

```toml
[tool.uv.sources]
myplugin = { path = "plugins/myplugin", develop = true }

[tool.uv.workspace]
members = [
    "agents-core",
    "plugins/myplugin",
    # ... other plugins
]
```