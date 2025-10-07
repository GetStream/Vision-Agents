# TTS Plugin Development Guide

## Overview

TTS (Text-to-Speech) plugins provide audio synthesis functionality for the vision-agents framework. They convert text into audio streams that can be played to users.

## Base Class

All TTS plugins should inherit from `vision_agents.core.tts.tts.TTS`:

```python
from vision_agents.core.tts.tts import TTS
from vision_agents.core.tts.events import TTSAudioEvent, TTSStartedEvent, TTSEndedEvent
from . import events

class MyTTS(TTS):
    def __init__(self, voice: str = "default"):
        super().__init__()
        # Register plugin-specific events
        self.events.register_events_from_module(events)
        self.voice = voice
```

## Event System Integration

TTS plugins emit and receive events to handle audio synthesis and coordinate with other components.

### Events TTS Plugins Emit

**Audio Events:**
- `TTSAudioEvent` - Generated audio data chunks
- `TTSSynthesisStartEvent` - Synthesis process started
- `TTSSynthesisCompleteEvent` - Synthesis process completed

**Error Events:**
- `TTSErrorEvent` - TTS processing errors
- `TTSConnectionEvent` - Connection state changes

**Custom Plugin Events:**
- `ElevenLabsAudioEvent` - ElevenLabs-specific audio events
- `CartesiaAudioEvent` - Cartesia-specific audio events
- `KokoroAudioEvent` - Kokoro-specific audio events

### Events TTS Plugins Receive

**From LLM:**
- `LLMResponseEvent` - Text to synthesize
- `LLMTextResponseDeltaEvent` - Streaming text chunks

**From Agent:**
- `AgentSayEvent` - Agent speech requests
- `AgentSayStartedEvent` - Agent speech started

**From VAD:**
- `VADSpeechStartEvent` - User starts speaking (pause TTS)
- `VADSpeechEndEvent` - User stops speaking (resume TTS)

### Event Flow Example

```python
# Typical TTS event flow
LLMResponseEvent → TTS Plugin → TTSAudioEvent → Audio Output
                       ↓
                 TTSSynthesisStartEvent → Agent
                       ↓
                 TTSSynthesisCompleteEvent → Agent

# Agent speech flow
AgentSayEvent → TTS Plugin → TTSAudioEvent → Audio Output
```

### Event Integration Patterns

**Simple TTS Plugin:**
```python
class MyTTS(TTS):
    async def synthesize(self, text: str) -> bytes:
        # Generate audio
        audio_data = await self._call_api(text)
        
        # Emit audio event
        self.events.send(TTSAudioEvent(
            plugin_name="mytts",
            audio_data=audio_data,
            sample_rate=24000
        ))
        
        return audio_data
```

**Advanced TTS Plugin with Custom Events:**
```python
class MyAdvancedTTS(TTS):
    def __init__(self):
        super().__init__()
        # Register custom events
        self.events.register_events_from_module(events)
    
    async def stream_synthesis(self, text: str):
        # Emit start event
        self.events.send(MyTTSStreamEvent(
            plugin_name="mytts",
            status="started",
            text=text
        ))
        
        async for chunk in self._stream_api(text):
            # Emit audio chunks
            self.events.send(TTSAudioEvent(
                plugin_name="mytts",
                audio_data=chunk.data,
                is_final_chunk=False
            ))
        
        # Emit completion event
        self.events.send(MyTTSStreamEvent(
            plugin_name="mytts",
            status="completed"
        ))
```

### Event Subscription

**Subscribe to LLM Events:**
```python
@self.events.subscribe
async def handle_llm_response(event: LLMResponseEvent):
    """Synthesize LLM responses."""
    audio = await self.synthesize(event.text)
```

**Subscribe to Agent Events:**
```python
@self.events.subscribe
async def handle_agent_say(event: AgentSayEvent):
    """Handle agent speech requests."""
    await self.synthesize(event.text)
```

**Subscribe to VAD Events:**
```python
@self.events.subscribe
async def handle_vad_events(event: VADSpeechStartEvent | VADSpeechEndEvent):
    """Control TTS based on user speech."""
    if isinstance(event, VADSpeechStartEvent):
        self.pause()  # Pause TTS when user speaks
    else:
        self.resume()  # Resume TTS when user stops
```

For detailed event system implementation, see [ai-events-example.md](ai-events-example.md).

## Required Methods

### synthesize

Implement the `synthesize` method for text-to-speech conversion:

```python
async def synthesize(self, text: str) -> bytes:
    """Convert text to audio bytes."""
    # Your TTS API implementation
    response = await self._call_tts_api(text)
    return response.audio_data
```

### Optional Methods

Implement additional methods as needed:

```python
async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
    """Stream audio synthesis for real-time playback."""
    async for chunk in self._stream_tts_api(text):
        yield chunk

def get_available_voices(self) -> List[str]:
    """Get list of available voices."""
    return self._get_voices_from_api()

def set_voice(self, voice: str) -> None:
    """Set the voice to use for synthesis."""
    self.voice = voice
```

## Real-time TTS

For real-time TTS with streaming support:

```python
class MyRealtimeTTS(TTS):
    def __init__(self, **kwargs):
        super().__init__()
        self.events.register_events_from_module(events)
        self._is_streaming = False
    
    async def start_stream(self, text: str):
        """Start streaming TTS synthesis."""
        self._is_streaming = True
        
        # Emit start event
        self.events.send(events.MyTTSStreamEvent(
            plugin_name="mytts",
            event_type="stream_started",
            event_data={"text": text}
        ))
        
        # Start streaming
        async for audio_chunk in self._stream_tts(text):
            if not self._is_streaming:
                break
                
            # Emit audio chunk
            self.events.send(events.MyTTSAudioEvent(
                plugin_name="mytts",
                audio_data=audio_chunk,
                sample_rate=24000,
                channels=1
            ))
    
    async def stop_stream(self):
        """Stop streaming TTS synthesis."""
        self._is_streaming = False
        
        # Emit stop event
        self.events.send(events.MyTTSStreamEvent(
            plugin_name="mytts",
            event_type="stream_stopped",
            event_data={}
        ))
```

## Example Implementation

Here's a complete example of a TTS plugin:

```python
# mytts/vision_agents/plugins/mytts/tts.py
import asyncio
from typing import Optional, List, AsyncIterator
from vision_agents.core.tts.tts import TTS
from vision_agents.core.tts.events import TTSAudioEvent
from . import events

class MyTTS(TTS):
    def __init__(self, voice: str = "default", api_key: Optional[str] = None):
        super().__init__()
        self.events.register_events_from_module(events)
        self.voice = voice
        self.api_key = api_key or os.getenv("MYTTS_API_KEY")
        self._is_streaming = False
    
    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes."""
        try:
            # Emit start event
            self.events.send(events.MyTTSStreamEvent(
                plugin_name="mytts",
                event_type="synthesis_started",
                event_data={"text": text, "voice": self.voice}
            ))
            
            # Call TTS API
            response = await self._call_api(text, voice=self.voice)
            
            # Emit audio event
            self.events.send(events.MyTTSAudioEvent(
                plugin_name="mytts",
                audio_data=response.audio_data,
                sample_rate=response.sample_rate,
                channels=response.channels
            ))
            
            # Emit completion event
            self.events.send(events.MyTTSStreamEvent(
                plugin_name="mytts",
                event_type="synthesis_completed",
                event_data={"duration": len(response.audio_data)}
            ))
            
            return response.audio_data
            
        except Exception as e:
            # Emit error event
            self.events.send(events.MyTTSErrorEvent(
                plugin_name="mytts",
                error_message=str(e),
                event_data=None
            ))
            raise
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """Stream audio synthesis for real-time playback."""
        self._is_streaming = True
        
        try:
            # Emit stream start event
            self.events.send(events.MyTTSStreamEvent(
                plugin_name="mytts",
                event_type="stream_started",
                event_data={"text": text}
            ))
            
            async for chunk in self._stream_api(text, voice=self.voice):
                if not self._is_streaming:
                    break
                    
                # Emit audio chunk
                self.events.send(events.MyTTSAudioEvent(
                    plugin_name="mytts",
                    audio_data=chunk.audio_data,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels
                ))
                
                yield chunk.audio_data
                
        finally:
            # Emit stream end event
            self.events.send(events.MyTTSStreamEvent(
                plugin_name="mytts",
                event_type="stream_ended",
                event_data={}
            ))
            self._is_streaming = False
    
    def stop_stream(self):
        """Stop streaming synthesis."""
        self._is_streaming = False
```

## Testing

Test your TTS plugin with events:

```python
import pytest
from mytts import MyTTS

@pytest.mark.asyncio
async def test_tts_events():
    tts = MyTTS()
    
    # Subscribe to events
    events_received = []
    
    @tts.events.subscribe
    async def handle_audio_event(event: MyTTSAudioEvent):
        events_received.append(event)
    
    @tts.events.subscribe
    async def handle_stream_event(event: MyTTSStreamEvent):
        events_received.append(event)
    
    # Synthesize text
    audio = await tts.synthesize("Hello world")
    
    # Wait for events to be processed
    await tts.events.wait()
    
    # Verify events were sent
    assert len(events_received) >= 3  # start, audio, completed
    assert any(e.event_type == "synthesis_started" for e in events_received)
    assert any(e.event_type == "synthesis_completed" for e in events_received)
    assert any(isinstance(e, MyTTSAudioEvent) for e in events_received)
```

## Best Practices

1. **Always register events**: Use `self.events.register_events_from_module(events)` in `__init__`
2. **Emit lifecycle events**: Send start, progress, and completion events
3. **Handle streaming**: Support both batch and streaming synthesis modes
4. **Error handling**: Emit error events when exceptions occur
5. **Audio metadata**: Include sample rate, channels, and other audio properties in events
6. **Voice management**: Support multiple voices and voice selection
7. **Performance**: Use streaming for real-time applications
8. **Testing**: Verify event flow and audio quality in tests
