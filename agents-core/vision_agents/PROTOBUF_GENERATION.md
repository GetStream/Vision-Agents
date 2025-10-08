# Protobuf Event Generation

## Overview

The `_generate_sfu_events.py` script automatically generates Python dataclass wrappers for protobuf messages from the SFU (Selective Forwarding Unit) event system. These generated classes inherit from `BaseEvent` and provide type-safe access to protobuf fields with all fields being optional.

## Location

- **Generator Script**: `agents-core/vision_agents/_generate_sfu_events.py`
- **Generated Output**: `agents-core/vision_agents/core/edge/sfu_events.py`

## Key Features

### 1. BaseEvent Inheritance

All generated classes inherit from `BaseEvent`, providing:
- `type`: Event type identifier (auto-set from protobuf full name)
- `event_id`: Unique identifier (auto-generated UUID)
- `timestamp`: Event creation time (auto-generated)
- `session_id`: Optional session identifier
- `user_metadata`: Optional user metadata

### 2. Optional Fields

All fields are optional, allowing event creation without a payload:

```python
event = AudioLevelEvent()  # All fields are optional
```

### 3. Advanced Type Mapping

The generator uses `_get_python_type_from_protobuf_field()` to map protobuf types to Python types with **full nested message type resolution**:

- Protobuf scalar types → Python primitives (int, float, str, bool, bytes)
- Protobuf repeated fields → `Optional[List[T]]`
- **Protobuf message types → Proper typed dataclass wrappers** (e.g., `Optional[Participant]`)
- Protobuf enum types → `Optional[int]`

All types are wrapped in `Optional` for flexibility.

#### Message Type Wrappers

The generator automatically creates dataclass wrappers for all protobuf message types used in events. These wrappers:
- Are placed at the top of the generated file
- Include all fields with proper Python types
- Support nested message types recursively
- Provide `from_proto()` class method for conversion
- Are fully typed for IDE autocomplete and type checking

Example:
```python
@dataclass
class Participant(DataClassJsonMixin):
    """Wrapper for stream.video.sfu.models.Participant."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    name: Optional[str] = None
    is_speaking: Optional[bool] = None
    audio_level: Optional[float] = None
    # ... all other fields
    
    @classmethod
    def from_proto(cls, proto_obj) -> 'Participant':
        """Create from protobuf Participant."""
        # ... conversion logic
```

### 4. Property-Based Access

Protobuf fields are exposed as properties with proper type hints:

```python
@property
def user_id(self) -> Optional[str]:
    """Access user_id field from the protobuf payload."""
    if self.payload is None:
        return None
    return getattr(self.payload, 'user_id', None)
```

### 5. Protobuf Integration

Each generated class provides:
- `from_proto(proto_obj)`: Create event from protobuf message
- `as_dict()`: Convert protobuf payload to dictionary
- `__getattr__()`: Delegate attribute access to protobuf payload

## Usage

### Regenerating Events

```bash
cd agents-core
uv run python vision_agents/_generate_sfu_events.py
```

### Verification

Verify type mappings and generated classes:

```bash
# Show type mappings
uv run python vision_agents/_generate_sfu_events.py --verify-types

# Verify generated classes
uv run python vision_agents/_generate_sfu_events.py --verify

# Both
uv run python vision_agents/_generate_sfu_events.py --verify-types --verify
```

### Example Usage

```python
from vision_agents.core.edge.sfu_events import (
    AudioLevelEvent,
    TrackUnpublishedEvent,
    Participant  # Now properly typed!
)
from getstream.video.rtc.pb.stream.video.sfu.event import events_pb2
from getstream.video.rtc.pb.stream.video.sfu.models import models_pb2

# Example 1: Simple event without payload
event1 = AudioLevelEvent()
print(event1.user_id)  # None

# Example 2: Event from protobuf
proto = events_pb2.AudioLevel(user_id='user123', level=0.85, is_speaking=True)
event2 = AudioLevelEvent.from_proto(proto)
print(event2.user_id)        # 'user123'
print(event2.level)          # 0.85
print(event2.is_speaking)    # True
print(event2.as_dict())      # {'user_id': 'user123', 'level': 0.85, 'is_speaking': True}

# Example 3: Event with nested message type (Participant)
proto_participant = models_pb2.Participant(
    user_id='user456',
    name='John Doe',
    is_speaking=True,
    audio_level=0.92
)
proto_track = events_pb2.TrackUnpublished(
    user_id='user456',
    participant=proto_participant
)
event3 = TrackUnpublishedEvent.from_proto(proto_track)

# Participant is properly typed as Participant dataclass!
participant: Participant = event3.participant  # Type-safe!
print(participant.user_id)     # 'user456'
print(participant.name)        # 'John Doe'
print(participant.is_speaking) # True
print(participant.audio_level) # 0.92

# IDE autocomplete works perfectly for all Participant fields!
```

## Verification Functions

### `_get_python_type_from_protobuf_field(field_descriptor)`

Determines the appropriate Python type annotation from a protobuf field descriptor. Maps protobuf types to Python types with proper handling of:
- Scalar types (int, float, str, bool, bytes)
- Repeated fields (lists)
- Message types (nested protobuf messages)
- Enum types

### `verify_field_types()`

Displays a comprehensive report of all field type mappings for verification:
```
AudioLevelEvent (AudioLevel):
  Protobuf type: stream.video.sfu.event.AudioLevel
  - user_id: type=9 (required) → Optional[str]
  - level: type=2 (required) → Optional[float]
  - is_speaking: type=8 (required) → Optional[bool]
```

### `verify_generated_classes()`

Verifies that generated classes match protobuf definitions by checking:
- Class exists in generated module
- All protobuf fields are accessible as properties
- Properties have correct types
- No missing or incorrect field mappings

## Generated Class Structure

Each generated class follows this pattern:

```python
@dataclass
class AudioLevelEvent(BaseEvent):
    """Dataclass event for video.sfu.event.events_pb2.AudioLevel."""
    type: str = field(default="stream.video.sfu.event.AudioLevel", init=False)
    payload: Optional[events_pb2.AudioLevel] = field(default=None, repr=False)

    @property
    def user_id(self) -> Optional[str]:
        """Access user_id field from the protobuf payload."""
        if self.payload is None:
            return None
        return getattr(self.payload, 'user_id', None)

    # ... more properties ...

    @classmethod
    def from_proto(cls, proto_obj: events_pb2.AudioLevel, **extra):
        """Create event instance from protobuf message."""
        return cls(payload=proto_obj, **extra)

    def as_dict(self) -> Dict[str, Any]:
        """Convert protobuf payload to dictionary."""
        if self.payload is None:
            return {}
        return _to_dict(self.payload)

    def __getattr__(self, item: str):
        """Delegate attribute access to protobuf payload."""
        if self.payload is not None:
            return getattr(self.payload, item)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")
```

## Import Strategy

The edge module uses absolute imports instead of relative imports to avoid naming conflicts with standard library modules (specifically avoiding conflicts with Python's `types` module).

```python
# In edge/__init__.py
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge import sfu_events
```

## Event Manager Integration

The EventManager has been updated to seamlessly handle the new protobuf events:

### How It Works

1. **Register protobuf event classes** like any other event:
   ```python
   from vision_agents.core.events.manager import EventManager
   from vision_agents.core.edge.sfu_events import AudioLevelEvent
   
   manager = EventManager()
   manager.register(AudioLevelEvent)
   ```

2. **Send events** in three ways:
   - Send wrapped events (already BaseEvent):
     ```python
     proto = events_pb2.AudioLevel(user_id='user123', level=0.85)
     event = AudioLevelEvent.from_proto(proto, session_id='session123')
     manager.send(event)  # BaseEvent fields preserved
     ```
   
   - Send raw protobuf messages (auto-wrapped):
     ```python
     proto = events_pb2.AudioLevel(user_id='user456', level=0.95)
     manager.send(proto)  # Automatically wrapped in AudioLevelEvent
     ```
   
   - Create events without payload (all fields optional):
     ```python
     event = AudioLevelEvent()  # No protobuf payload needed
     manager.send(event)
     ```

3. **Subscribe to protobuf events** like any other event:
   ```python
   @manager.subscribe
   async def handle_audio(event: AudioLevelEvent):
       print(f"User: {event.user_id}, Level: {event.level}")
       print(f"Session: {event.session_id}, ID: {event.event_id}")
   ```

### Key Improvements

- **No double-wrapping**: Already-wrapped BaseEvent subclasses are not re-wrapped
- **BaseEvent fields preserved**: session_id, event_id, timestamp all work correctly
- **Simplified logic**: Single check distinguishes raw protobuf from wrapped events
- **Type safety**: All generated events properly inherit from BaseEvent
- **Flexible usage**: Use raw protobuf or wrapped events interchangeably

