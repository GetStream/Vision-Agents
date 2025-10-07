# STT Plugin Development Guide

## Overview

STT (Speech-to-Text) plugins provide audio transcription functionality for the vision-agents framework. They convert audio streams into text transcripts that can be processed by other components.

## Base Class

All STT plugins should inherit from `vision_agents.core.stt.stt.STT`:

```python
from vision_agents.core.stt.stt import STT
from vision_agents.core.stt.events import STTTranscriptEvent, STTStartedEvent, STTEndedEvent
from . import events

class MySTT(STT):
    def __init__(self, model: str = "default"):
        super().__init__()
        # Register plugin-specific events
        self.events.register_events_from_module(events)
        self.model = model
```

## Event System Integration

STT plugins emit and receive events to handle speech transcription and coordinate with other components.

### Events STT Plugins Emit

**Transcript Events:**
- `STTTranscriptEvent` - Complete speech transcripts
- `STTPartialTranscriptEvent` - Partial/streaming transcripts
- `STTConnectionEvent` - Connection state changes

**Error Events:**
- `STTErrorEvent` - STT processing errors

**Custom Plugin Events:**
- `DeepgramTranscriptEvent` - Deepgram-specific events
- `MoonshineTranscriptEvent` - Moonshine-specific events
- `FalTranscriptEvent` - FAL-specific events

### Events STT Plugins Receive

**From Audio Input:**
- `AudioReceivedEvent` - Raw audio data from microphone
- `VADAudioEvent` - Processed audio segments from VAD

**From VAD:**
- `VADSpeechStartEvent` - User starts speaking
- `VADSpeechEndEvent` - User stops speaking (trigger final transcription)

**From Agent:**
- Custom agent events for transcription control

### Event Flow Example

```python
# Typical STT event flow
AudioReceivedEvent → STT Plugin → STTTranscriptEvent → LLM Plugin
                         ↓
                   STTPartialTranscriptEvent → Agent (for real-time feedback)

# VAD-integrated flow
VADSpeechEndEvent → STT Plugin → STTTranscriptEvent → LLM Plugin
```

### Event Integration Patterns

**Simple STT Plugin:**
```python
class MySTT(STT):
    async def transcribe(self, audio_data: bytes) -> str:
        # Transcribe audio
        result = await self._call_api(audio_data)
        
        # Emit transcript event
        self.events.send(STTTranscriptEvent(
            plugin_name="mystt",
            text=result.text,
            confidence=result.confidence,
            is_final=True
        ))
        
        return result.text
```

**Advanced STT Plugin with Streaming:**
```python
class MyAdvancedSTT(STT):
    def __init__(self):
        super().__init__()
        # Register custom events
        self.events.register_events_from_module(events)
    
    async def stream_transcribe(self, audio_stream):
        # Emit start event
        self.events.send(MySTTStreamEvent(
            plugin_name="mystt",
            status="started"
        ))
        
        async for chunk in self._stream_api(audio_stream):
            # Emit partial transcripts
            self.events.send(STTPartialTranscriptEvent(
                plugin_name="mystt",
                text=chunk.text,
                is_final=False
            ))
        
        # Emit final transcript
        self.events.send(STTTranscriptEvent(
            plugin_name="mystt",
            text=final_text,
            is_final=True
        ))
```

### Event Subscription

**Subscribe to Audio Events:**
```python
@self.events.subscribe
async def handle_audio_input(event: AudioReceivedEvent):
    """Process incoming audio for transcription."""
    if event.pcm_data:
        transcript = await self.transcribe(event.pcm_data)
```

**Subscribe to VAD Events:**
```python
@self.events.subscribe
async def handle_vad_end(event: VADSpeechEndEvent):
    """Process final transcript when user stops speaking."""
    if self._pending_audio:
        transcript = await self.transcribe_final(self._pending_audio)
        self._pending_audio = None
```

**Subscribe to Agent Events:**
```python
@self.events.subscribe
async def handle_agent_control(event: AgentControlEvent):
    """Handle agent control commands."""
    if event.command == "pause_transcription":
        self.pause()
    elif event.command == "resume_transcription":
        self.resume()
```

For detailed event system implementation, see [ai-events-example.md](ai-events-example.md).

## Required Methods

### transcribe

Implement the `transcribe` method for audio-to-text conversion:

```python
async def transcribe(self, audio_data: bytes) -> str:
    """Convert audio bytes to text."""
    # Your STT API implementation
    result = await self._call_stt_api(audio_data)
    return result.text
```

### Optional Methods

Implement additional methods as needed:

```python
async def transcribe_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Stream audio transcription for real-time processing."""
    async for audio_chunk in audio_stream:
        transcript = await self.transcribe(audio_chunk)
        if transcript:
            yield transcript

def get_supported_languages(self) -> List[str]:
    """Get list of supported languages."""
    return self._get_languages_from_api()

def set_language(self, language: str) -> None:
    """Set the language for transcription."""
    self.language = language
```

## Real-time STT

For real-time STT with streaming support:

```python
class MyRealtimeSTT(STT):
    def __init__(self, **kwargs):
        super().__init__()
        self.events.register_events_from_module(events)
        self._is_streaming = False
        self._audio_buffer = []
    
    async def start_stream(self):
        """Start streaming STT transcription."""
        self._is_streaming = True
        self._audio_buffer = []
        
        # Emit start event
        self.events.send(events.MySTTStreamEvent(
            plugin_name="mystt",
            event_type="stream_started",
            event_data={}
        ))
    
    async def process_audio_chunk(self, audio_chunk: bytes):
        """Process a chunk of audio data."""
        if not self._is_streaming:
            return
            
        self._audio_buffer.append(audio_chunk)
        
        # Process for interim results
        interim_result = await self._get_interim_result(audio_chunk)
        if interim_result:
            # Emit interim transcript
            self.events.send(events.MySTTTranscriptEvent(
                plugin_name="mystt",
                text=interim_result.text,
                confidence=interim_result.confidence,
                is_final=False
            ))
    
    async def stop_stream(self) -> str:
        """Stop streaming and return final transcript."""
        self._is_streaming = False
        
        # Process final audio buffer
        final_audio = b''.join(self._audio_buffer)
        final_result = await self.transcribe(final_audio)
        
        # Emit final transcript
        self.events.send(events.MySTTTranscriptEvent(
            plugin_name="mystt",
            text=final_result,
            confidence=1.0,
            is_final=True
        ))
        
        # Emit stop event
        self.events.send(events.MySTTStreamEvent(
            plugin_name="mystt",
            event_type="stream_stopped",
            event_data={"final_text": final_result}
        ))
        
        return final_result
```

## VAD Integration

Integrate with Voice Activity Detection (VAD) for better transcription timing:

```python
class MyVADIntegratedSTT(STT):
    def __init__(self, **kwargs):
        super().__init__()
        self.events.register_events_from_module(events)
        self._audio_buffer = []
        self._is_speaking = False
    
    @self.events.subscribe
    async def handle_vad_start(self, event: VADSpeechStartEvent):
        """Handle speech start detection."""
        self._is_speaking = True
        self._audio_buffer = []
        
        # Emit start event
        self.events.send(events.MySTTStreamEvent(
            plugin_name="mystt",
            event_type="speech_started",
            event_data={"speech_probability": event.speech_probability}
        ))
    
    @self.events.subscribe
    async def handle_vad_end(self, event: VADSpeechEndEvent):
        """Handle speech end detection."""
        if not self._is_speaking:
            return
            
        self._is_speaking = False
        
        # Process accumulated audio
        if self._audio_buffer:
            final_audio = b''.join(self._audio_buffer)
            transcript = await self.transcribe(final_audio)
            
            # Emit final transcript
            self.events.send(events.MySTTTranscriptEvent(
                plugin_name="mystt",
                text=transcript,
                confidence=1.0,
                is_final=True
            ))
        
        # Emit end event
        self.events.send(events.MySTTStreamEvent(
            plugin_name="mystt",
            event_type="speech_ended",
            event_data={"duration_ms": event.total_speech_duration_ms}
        ))
    
    async def process_audio(self, audio_data: bytes):
        """Process audio data during speech."""
        if self._is_speaking:
            self._audio_buffer.append(audio_data)
            
            # Process for interim results
            interim_result = await self._get_interim_result(audio_data)
            if interim_result:
                self.events.send(events.MySTTTranscriptEvent(
                    plugin_name="mystt",
                    text=interim_result.text,
                    confidence=interim_result.confidence,
                    is_final=False
                ))
```

## Example Implementation

Here's a complete example of an STT plugin:

```python
# mystt/vision_agents/plugins/mystt/stt.py
import asyncio
from typing import Optional, List, AsyncIterator
from vision_agents.core.stt.stt import STT
from vision_agents.core.stt.events import STTTranscriptEvent
from . import events

class MySTT(STT):
    def __init__(self, model: str = "default", api_key: Optional[str] = None):
        super().__init__()
        self.events.register_events_from_module(events)
        self.model = model
        self.api_key = api_key or os.getenv("MYSTT_API_KEY")
        self._is_streaming = False
        self._audio_buffer = []
    
    async def transcribe(self, audio_data: bytes) -> str:
        """Convert audio bytes to text."""
        try:
            # Emit start event
            self.events.send(events.MySTTStreamEvent(
                plugin_name="mystt",
                event_type="transcription_started",
                event_data={"audio_length": len(audio_data), "model": self.model}
            ))
            
            # Call STT API
            result = await self._call_api(audio_data, model=self.model)
            
            # Emit transcript event
            self.events.send(events.MySTTTranscriptEvent(
                plugin_name="mystt",
                text=result.text,
                confidence=result.confidence,
                language=result.language,
                is_final=True
            ))
            
            # Emit completion event
            self.events.send(events.MySTTStreamEvent(
                plugin_name="mystt",
                event_type="transcription_completed",
                event_data={"text_length": len(result.text)}
            ))
            
            return result.text
            
        except Exception as e:
            # Emit error event
            self.events.send(events.MySTTErrorEvent(
                plugin_name="mystt",
                error_message=str(e),
                event_data=None
            ))
            raise
    
    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """Stream audio transcription for real-time processing."""
        self._is_streaming = True
        
        try:
            # Emit stream start event
            self.events.send(events.MySTTStreamEvent(
                plugin_name="mystt",
                event_type="stream_started",
                event_data={}
            ))
            
            async for audio_chunk in audio_stream:
                if not self._is_streaming:
                    break
                
                # Process chunk
                result = await self._process_chunk(audio_chunk)
                if result and result.text:
                    # Emit interim transcript
                    self.events.send(events.MySTTTranscriptEvent(
                        plugin_name="mystt",
                        text=result.text,
                        confidence=result.confidence,
                        is_final=result.is_final
                    ))
                    
                    yield result.text
                    
        finally:
            # Emit stream end event
            self.events.send(events.MySTTStreamEvent(
                plugin_name="mystt",
                event_type="stream_ended",
                event_data={}
            ))
            self._is_streaming = False
```

## Testing

Test your STT plugin with events:

```python
import pytest
from mystt import MySTT

@pytest.mark.asyncio
async def test_stt_events():
    stt = MySTT()
    
    # Subscribe to events
    events_received = []
    
    @stt.events.subscribe
    async def handle_transcript_event(event: MySTTTranscriptEvent):
        events_received.append(event)
    
    @stt.events.subscribe
    async def handle_stream_event(event: MySTTStreamEvent):
        events_received.append(event)
    
    # Transcribe audio
    transcript = await stt.transcribe(b"fake_audio_data")
    
    # Wait for events to be processed
    await stt.events.wait()
    
    # Verify events were sent
    assert len(events_received) >= 3  # start, transcript, completed
    assert any(e.event_type == "transcription_started" for e in events_received)
    assert any(e.event_type == "transcription_completed" for e in events_received)
    assert any(isinstance(e, MySTTTranscriptEvent) for e in events_received)
```

## Best Practices

1. **Always register events**: Use `self.events.register_events_from_module(events)` in `__init__`
2. **Emit lifecycle events**: Send start, progress, and completion events
3. **Support streaming**: Provide both batch and streaming transcription modes
4. **Handle interim results**: Emit partial transcripts for real-time feedback
5. **VAD integration**: Work well with Voice Activity Detection
6. **Error handling**: Emit error events when exceptions occur
7. **Confidence scores**: Include confidence levels in transcript events
8. **Language support**: Support multiple languages and language detection
9. **Performance**: Optimize for low latency in real-time applications
10. **Testing**: Verify event flow and transcription accuracy in tests
