# LLM Plugin Development Guide

## Overview

LLM plugins provide language model functionality for the vision-agents framework. They handle text generation, conversation management, and function calling capabilities.

## Base Class

All LLM plugins should inherit from `vision_agents.core.llm.llm.LLM`:

```python
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.types import LLMTextResponseDeltaEvent
from . import events

class MyLLM(LLM):
    def __init__(self, model: str = "my-model"):
        super().__init__()
        # Register plugin-specific events
        self.events.register_events_from_module(events)
        self.model = model
```

## Event System Integration

LLM plugins emit and receive events to communicate with other components in the system.

### Events LLM Plugins Emit

**Response Events:**
- `LLMResponseEvent` - Complete LLM responses
- `LLMTextResponseDeltaEvent` - Streaming text chunks
- `LLMTextResponseCompletedEvent` - Response completion

**Custom Plugin Events:**
- `OpenAIStreamEvent` - OpenAI-specific streaming events
- `GeminiResponseEvent` - Gemini response chunks
- `ClaudeStreamEvent` - Claude streaming events
- `XAIChunkEvent` - xAI chunk events

**Error Events:**
- `LLMErrorEvent` - LLM processing errors
- Plugin-specific error events

### Events LLM Plugins Receive

**From STT:**
- `STTTranscriptEvent` - User speech transcripts
- `STTPartialTranscriptEvent` - Partial transcripts

**From VAD:**
- `VADSpeechStartEvent` - User starts speaking
- `VADSpeechEndEvent` - User stops speaking

**From Agent:**
- `AgentSayEvent` - Agent wants to speak
- Custom agent events

### Event Flow Example

```python
# Typical LLM event flow
STTTranscriptEvent → LLM Plugin → LLMResponseEvent → TTS Plugin
                         ↓
                   LLMTextResponseDeltaEvent → Agent
                         ↓
                   LLMTextResponseCompletedEvent → Agent
```

### Event Integration Patterns

**Simple LLM Plugin:**
```python
class MyLLM(LLM):
    async def simple_response(self, text):
        # Generate response
        response = await self._call_api(text)
        
        # Emit standardized events
        self.events.send(LLMResponseEvent(
            plugin_name="myllm",
            response=response,
            text=response.text
        ))
        
        return response
```

**Advanced LLM Plugin with Custom Events:**
```python
class MyAdvancedLLM(LLM):
    def __init__(self):
        super().__init__()
        # Register custom events
        self.events.register_events_from_module(events)
    
    async def stream_response(self, text):
        # Emit custom streaming events
        self.events.send(MyLLMStreamEvent(
            plugin_name="myllm",
            status="started"
        ))
        
        async for chunk in self._stream_api(text):
            # Emit delta events
            self.events.send(LLMTextResponseDeltaEvent(
                plugin_name="myllm",
                delta=chunk.text
            ))
        
        # Emit completion event
        self.events.send(MyLLMStreamEvent(
            plugin_name="myllm",
            status="completed"
        ))
```

### Event Subscription

**Subscribe to Input Events:**
```python
@self.events.subscribe
async def handle_user_input(event: STTTranscriptEvent):
    """Process user speech input."""
    response = await self.simple_response(event.text)
```

**Subscribe to Agent Events:**
```python
@self.events.subscribe
async def handle_agent_say(event: AgentSayEvent):
    """Handle when agent wants to speak."""
    # Process agent speech request
    pass
```

For detailed event system implementation, see [ai-events-example.md](ai-events-example.md).

## Required Methods

### simple_response

Implement the `simple_response` method for basic text generation:

```python
async def simple_response(
    self,
    text: str,
    processors: Optional[List[BaseProcessor]] = None,
    participant: Optional[Participant] = None,
) -> LLMResponseEvent[Any]:
    """Generate a simple response to the given text."""
    # Your implementation here
    response = await self._call_llm_api(text)
    return LLMResponseEvent(response, response.text)
```

### Function Calling Support

Implement function calling methods if supported:

```python
def _convert_tools_to_provider_format(self, tools: List[ToolSchema]) -> List[Dict[str, Any]]:
    """Convert tools to your provider's format."""
    # Convert vision-agents ToolSchema to your provider's format
    return converted_tools

def _extract_tool_calls_from_response(self, response: Any) -> List[NormalizedToolCallItem]:
    """Extract tool calls from provider response."""
    # Extract and normalize tool calls from your provider's response
    return normalized_tool_calls
```

## Realtime/STS Support

For Speech-to-Speech (STS) functionality, inherit from `vision_agents.core.llm.realtime.Realtime`:

```python
from vision_agents.core.llm.realtime import Realtime

class MyRealtimeLLM(Realtime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Register events
        self.events.register_events_from_module(events)
    
    async def connect(self):
        """Connect to the realtime service."""
        # Your connection logic
        pass
    
    async def disconnect(self):
        """Disconnect from the realtime service."""
        # Your disconnection logic
        pass
```

## Example Implementation

Here's a complete example of an LLM plugin:

```python
# myllm/vision_agents/plugins/myllm/llm.py
from typing import Optional, List, Any
from vision_agents.core.llm.llm import LLM, LLMResponseEvent
from vision_agents.core.llm.types import LLMTextResponseDeltaEvent
from vision_agents.core.processors import Processor
from . import events


class MyLLM(LLM):
    def __init__(self, model: str = "my-model", api_key: Optional[str] = None):
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.api_key = api_key or os.getenv("MYLLM_API_KEY")

    async def simple_response(
            self,
            text: str,
            processors: Optional[List[Processor]] = None,
            participant: Optional[Participant] = None,
    ) -> LLMResponseEvent[Any]:
        """Generate a response using the LLM API."""
        try:
            # Call your LLM API
            response = await self._call_api(text)

            # Emit events
            self.events.send(events.MyLLMStreamEvent(
                plugin_name="myllm",
                event_type="response",
                event_data=response
            ))

            return LLMResponseEvent(response, response.text)

        except Exception as e:
            # Emit error event
            self.events.send(events.MyLLMErrorEvent(
                plugin_name="myllm",
                error_message=str(e),
                event_data=None
            ))
            raise
```

## Testing

Test your LLM plugin with events:

```python
import pytest
from myllm import MyLLM

@pytest.mark.asyncio
async def test_llm_events():
    llm = MyLLM()
    
    # Subscribe to events
    events_received = []
    
    @llm.events.subscribe
    async def handle_stream_event(event: MyLLMStreamEvent):
        events_received.append(event)
    
    # Generate response
    response = await llm.simple_response("Hello")
    
    # Wait for events to be processed
    await llm.events.wait()
    
    # Verify events were sent
    assert len(events_received) > 0
    assert events_received[0].event_type == "response"
```

## Best Practices

1. **Always register events**: Use `self.events.register_events_from_module(events)` in `__init__`
2. **Emit standardized events**: Send both raw and standardized events for compatibility
3. **Handle errors gracefully**: Emit error events when exceptions occur
4. **Use type hints**: Properly type your event handlers for better IDE support
5. **Test event flow**: Verify that events are sent and received correctly
6. **Document events**: Clearly document what each event represents and when it's sent
