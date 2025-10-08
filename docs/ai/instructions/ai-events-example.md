# Event System Developer Guide

This guide explains how to use the event system when building plugins for vision-agents. The event system provides a centralized way to handle asynchronous communication between components.

## Table of Contents

1. [Event Manager Overview](#event-manager-overview)
2. [Creating Event Managers](#creating-event-managers)
3. [Event Registration](#event-registration)
4. [Event Definition](#event-definition)
5. [Sending Events](#sending-events)
6. [Subscribing to Events](#subscribing-to-events)
7. [Plugin Integration Patterns](#plugin-integration-patterns)
8. [Best Practices](#best-practices)
9. [Real Plugin Examples](#real-plugin-examples)

## Event Manager Overview

The `EventManager` is the core component that handles event registration, subscription, and processing. It provides:

- **Event Registration**: Register custom event types
- **Event Subscription**: Subscribe to events with type hints
- **Asynchronous Processing**: Background event processing
- **Error Handling**: Automatic exception event generation
- **Module Registration**: Bulk event registration from modules

## Creating Event Managers

### Basic Event Manager

```python
from vision_agents.core.events.manager import EventManager

# Create a basic event manager
manager = EventManager()

# Create with custom settings
manager = EventManager(ignore_unknown_events=False)  # Raise errors for unknown events
```

### Event Manager in Plugins

All core components (LLM, TTS, STT, VAD) automatically provide an `EventManager` instance:

```python
class MyPlugin:
    def __init__(self):
        # EventManager is automatically available
        self.events = EventManager()
        
        # Register your custom events
        self.events.register_events_from_module(events)
```

## Event Registration

### Register Individual Events

```python
from vision_agents.core.events.manager import EventManager
from my_plugin.events import MyCustomEvent

manager = EventManager()
manager.register(MyCustomEvent)
```

### Register Events from Module

```python
# Register all events from a module
from my_plugin import events

manager = EventManager()
manager.register_events_from_module(events)
```

### Plugin Event Registration Pattern

```python
class MyPlugin:
    def __init__(self):
        super().__init__()
        # Register plugin-specific events
        self.events.register_events_from_module(events)
```

## Event Definition

### Basic Event Structure

All events must inherit from `PluginBaseEvent` and follow this pattern:

```python
from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent
from typing import Optional, Any

@dataclass
class MyPluginEvent(PluginBaseEvent):
    """Event emitted when something happens in my plugin."""
    type: str = field(default='plugin.myplugin.event', init=False)
    data: Optional[Any] = None
    message: Optional[str] = None
    metadata: Optional[dict] = None
```

### Event Type Naming Convention

- **Format**: `plugin.{plugin_name}.{event_name}`
- **Examples**:
  - `plugin.openai.stream`
  - `plugin.gemini.connected`
  - `plugin.tts.audio`
  - `agent.say`

### Event Validation

```python
@dataclass
class MyValidatedEvent(PluginBaseEvent):
    type: str = field(default='plugin.myplugin.validated', init=False)
    text: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.text:
            raise ValueError("Text cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
```

## Sending Events

### Basic Event Sending

```python
# Send a single event
self.events.send(MyPluginEvent(
    plugin_name="myplugin",
    data={"key": "value"},
    message="Something happened"
))

# Send multiple events
self.events.send(
    MyPluginEvent(plugin_name="myplugin", message="Event 1"),
    MyPluginEvent(plugin_name="myplugin", message="Event 2")
)
```

### Event Sending in Plugins

```python
class MyPlugin:
    async def process_data(self, data):
        try:
            # Emit start event
            self.events.send(MyPluginEvent(
                plugin_name="myplugin",
                message="Processing started",
                data=data
            ))
            
            # Do processing
            result = await self._process(data)
            
            # Emit success event
            self.events.send(MyPluginEvent(
                plugin_name="myplugin",
                message="Processing completed",
                data=result
            ))
            
        except Exception as e:
            # Emit error event
            self.events.send(MyPluginErrorEvent(
                plugin_name="myplugin",
                error_message=str(e),
                error=e
            ))
            raise
```

## Subscribing to Events

### Basic Event Subscription

```python
# Subscribe to specific event type
@self.events.subscribe
async def handle_my_event(event: MyPluginEvent):
    print(f"Received: {event.message}")

# Subscribe to multiple event types using Union
@self.events.subscribe
async def handle_multiple_events(event: MyPluginEvent | OtherPluginEvent):
    print(f"Received event: {event.type}")
```

### Event Subscription Patterns

```python
class MyPlugin:
    def __init__(self):
        super().__init__()
        self.events.register_events_from_module(events)
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the plugin."""
        
        @self.events.subscribe
        async def handle_stt_transcript(event: STTTranscriptEvent):
            """Handle speech-to-text transcripts."""
            if event.is_final:
                await self._process_transcript(event.text)
        
        @self.events.subscribe
        async def handle_llm_response(event: LLMResponseEvent):
            """Handle LLM responses."""
            await self._process_llm_response(event.text)
        
        @self.events.subscribe
        async def handle_error_events(event: MyPluginErrorEvent | STTErrorEvent):
            """Handle error events."""
            self.logger.error(f"Error in {event.__class__.__name__}: {event.error_message}")
```

### Cross-Plugin Event Handling

```python
# Subscribe to events from other plugins
@agent.events.subscribe
async def handle_openai_events(event: OpenAIStreamEvent):
    print(f"OpenAI event: {event.event_type}")

@agent.events.subscribe
async def handle_gemini_events(event: GeminiConnectedEvent):
    print(f"Gemini connected: {event.model}")
```

## Plugin Integration Patterns

### Pattern 1: Plugin with Custom Events

```python
# my_plugin/events.py
from dataclasses import dataclass, field
from vision_agents.core.events import PluginBaseEvent

@dataclass
class MyPluginStartEvent(PluginBaseEvent):
    type: str = field(default='plugin.myplugin.start', init=False)
    config: Optional[dict] = None

@dataclass
class MyPluginDataEvent(PluginBaseEvent):
    type: str = field(default='plugin.myplugin.data', init=False)
    data: Optional[bytes] = None
    metadata: Optional[dict] = None

# my_plugin/plugin.py
from vision_agents.core.plugin_base import PluginBase
from . import events

class MyPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        # Register custom events
        self.events.register_events_from_module(events)
    
    async def process(self, data):
        # Send custom events
        self.events.send(events.MyPluginStartEvent(
            plugin_name="myplugin",
            config=self.config
        ))
        
        result = await self._process_data(data)
        
        self.events.send(events.MyPluginDataEvent(
            plugin_name="myplugin",
            data=result,
            metadata={"processed_at": time.time()}
        ))
```

### Pattern 2: Plugin with Base Events Only

```python
# Simple plugin using base class events
from vision_agents.core.stt.stt import STT
from vision_agents.core.stt.events import STTTranscriptEvent, STTErrorEvent

class SimpleSTT(STT):
    def __init__(self):
        super().__init__()
        # No need to register custom events - use base class events
    
    async def transcribe(self, audio_data: bytes) -> str:
        try:
            result = await self._call_api(audio_data)
            
            # Send base class event
            self.events.send(STTTranscriptEvent(
                plugin_name="simple_stt",
                text=result.text,
                confidence=result.confidence,
                is_final=True
            ))
            
            return result.text
            
        except Exception as e:
            # Send error event
            self.events.send(STTErrorEvent(
                plugin_name="simple_stt",
                error=e,
                error_code="transcription_failed"
            ))
            raise
```

### Pattern 3: Agent Event Integration

```python
# Agent with event-driven architecture
from vision_agents.core.agents import Agent
from vision_agents.core.agents.events import AgentSayEvent

class MyAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_agent_handlers()
    
    def _setup_agent_handlers(self):
        @self.events.subscribe
        async def handle_agent_say(event: AgentSayEvent):
            """Handle when agent wants to say something."""
            print(f"Agent wants to say: {event.text}")
            
            # Process the speech request
            await self._process_speech_request(event)
    
    # Three ways to send events:
    
    # Method 1: Direct event sending
    def send_custom_event(self, data):
        self.events.send(MyCustomEvent(
            plugin_name="agent",
            data=data
        ))
    
    # Method 2: Convenience method
    def send_event_convenience(self, data):
        self.send(MyCustomEvent(
            plugin_name="agent",
            data=data
        ))
    
    # Method 3: High-level speech
    async def make_agent_speak(self, text):
        await self.say(text, metadata={"source": "custom_handler"})
```

## Best Practices

### 1. Event Naming and Structure

```python
# ✅ Good: Clear, descriptive event names
@dataclass
class TranscriptionCompletedEvent(PluginBaseEvent):
    type: str = field(default='plugin.stt.transcription_completed', init=False)
    text: str = ""
    confidence: float = 0.0
    duration_ms: float = 0.0

# ❌ Bad: Vague event names
@dataclass
class Event1(PluginBaseEvent):
    type: str = field(default='plugin.stt.event1', init=False)
    data: Any = None
```

### 2. Error Handling

```python
# Always emit error events when exceptions occur
try:
    result = await self._process_data(data)
    self.events.send(SuccessEvent(plugin_name="myplugin", result=result))
except Exception as e:
    self.events.send(ErrorEvent(
        plugin_name="myplugin",
        error_message=str(e),
        error=e
    ))
    raise
```

### 3. Event Metadata

```python
# Include relevant metadata for debugging and analytics
self.events.send(ProcessingEvent(
    plugin_name="myplugin",
    data=result,
    metadata={
        "processing_time_ms": duration,
        "input_size": len(input_data),
        "model_version": self.model_version,
        "timestamp": time.time()
    }
))
```

### 4. Event Validation

```python
@dataclass
class ValidatedEvent(PluginBaseEvent):
    type: str = field(default='plugin.myplugin.validated', init=False)
    text: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
```

### 5. Testing Events

```python
import pytest
from my_plugin import MyPlugin

@pytest.mark.asyncio
async def test_plugin_events():
    plugin = MyPlugin()
    
    # Track events
    events_received = []
    
    @plugin.events.subscribe
    async def track_events(event):
        events_received.append(event)
    
    # Trigger event
    await plugin.process_data("test")
    
    # Wait for events
    await plugin.events.wait()
    
    # Verify events
    assert len(events_received) > 0
    assert any(isinstance(e, MyPluginEvent) for e in events_received)
```

## Real Plugin Examples

### OpenAI Plugin Events

```python
# plugins/openai/vision_agents/plugins/openai/events.py
@dataclass
class OpenAIStreamEvent(PluginBaseEvent):
    type: str = field(default='plugin.openai.stream', init=False)
    event_type: Optional[str] = None
    event_data: Optional[Any] = None

@dataclass
class LLMErrorEvent(PluginBaseEvent):
    type: str = field(default='plugin.llm.error', init=False)
    error_message: Optional[str] = None
    event_data: Optional[Any] = None

# Usage in OpenAI LLM
class OpenAILLM(LLM):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
    
    def _standardize_and_emit_event(self, event: ResponseStreamEvent):
        # Send raw OpenAI event
        self.events.send(events.OpenAIStreamEvent(
            plugin_name="openai",
            event_type=event.type,
            event_data=event
        ))
        
        if event.type == "response.error":
            self.events.send(events.LLMErrorEvent(
                plugin_name="openai",
                error_message=getattr(event, "error", {}).get("message", "Unknown error"),
                event_data=event
            ))
```

### Gemini Plugin Events

```python
# plugins/gemini/vision_agents/plugins/gemini/events.py
@dataclass
class GeminiConnectedEvent(PluginBaseEvent):
    type: str = field(default='plugin.gemini.connected', init=False)
    model: Optional[str] = None

@dataclass
class GeminiAudioEvent(PluginBaseEvent):
    type: str = field(default='plugin.gemini.audio', init=False)
    audio_data: Optional[bytes] = None

# Usage in Gemini Realtime
class Realtime(realtime.Realtime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.events.register_events_from_module(events)
    
    async def connect(self):
        # ... connection logic ...
        
        # Emit connection event
        self.events.send(events.GeminiConnectedEvent(
            plugin_name="gemini",
            model=self.model
        ))
```

## Event System Summary

The event system provides a powerful, flexible way to build plugins with:

- **Centralized Communication**: All components communicate through events
- **Type Safety**: Event handlers use type hints for better IDE support
- **Error Handling**: Automatic exception event generation
- **Modularity**: Easy to add new event types and handlers
- **Testing**: Simple to test by subscribing to events
- **Monitoring**: Easy to add logging and analytics by subscribing to events

Use this guide when building new plugins to ensure consistent, maintainable, and testable code with the event system.
