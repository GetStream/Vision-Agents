# Debugging Event Subscription Issue

## The Problem

The error "Multiple seperated events per handler are not supported, use Union instead" occurs when trying to subscribe to `SquatCompletedEvent`.

## Root Cause

The event manager's `subscribe()` method (in `agents-core/vision_agents/core/events/manager.py`) uses `typing.get_type_hints(function)` to detect event types. This function returns ALL type hints from a function, including:
- Parameter type hints
- Return type hints  
- Potentially closure variables if they have type hints

When the handler function is defined inside `create_agent()`, Python's type system may be detecting multiple things that look like event types (anything with a `type` attribute).

## How to Debug

1. **Inspect what type hints are being detected:**

Add this debug code before `agent.events.subscribe(handler)`:

```python
import typing
from typing import get_type_hints

# Debug: See what type hints are detected
annotations = get_type_hints(handler)
print("DEBUG: Handler annotations:", annotations)
for name, hint in annotations.items():
    print(f"  {name}: {hint}")
    if hasattr(hint, 'type'):
        print(f"    -> Has 'type' attribute: {getattr(hint, 'type', None)}")
```

2. **Check if the event is properly registered:**

```python
# Debug: Check if event is registered
print("DEBUG: Registered events:", list(agent.events._events.keys()))
print("DEBUG: SquatCompletedEvent.type:", SquatCompletedEvent.type)
print("DEBUG: Is registered?", SquatCompletedEvent.type in agent.events._events)
```

3. **Try a minimal handler outside the function:**

```python
# At module level (outside create_agent)
async def minimal_handler(event: SquatCompletedEvent) -> None:
    logger.info(f"Squat completed: {event.rep_count}")

# In create_agent, after registering:
agent.events.subscribe(minimal_handler)
```

4. **Check the event manager code:**

Look at `agents-core/vision_agents/core/events/manager.py` line 334-373. The `subscribe()` method:
- Gets all type hints with `typing.get_type_hints(function)`
- Iterates through ALL annotations (line 339)
- Checks if each has a `type` attribute (line 350)
- Throws error if it finds multiple event types (line 352-355)

## Potential Solutions

1. **Define handler at module level** - This avoids closure type hint issues
2. **Use a class-based handler** - Classes might avoid the type hint detection issue
3. **Manually register handler** - Bypass the subscribe method and manually add to `_handlers`
4. **Fix the event manager** - Modify it to only check parameter annotations, not all annotations

## Current Workaround

Events are being emitted from the processor (see `squat_yolo_processor.py` line 101), but the subscription is disabled. The events are still being sent, they're just not being handled yet.

To temporarily test, you could manually handle events in the processor or use a different event handling mechanism.

