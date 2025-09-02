# Function Calling System - RFC

## Overview

This document describes the design and implementation of the function calling system for Stream Agents' LLM classes.

## Design Principles

### 1. Decorator-Based Registration
Functions are registered using a simple decorator pattern:

```python
@llm.function("Get weather for a location")
async def get_weather(location: str, unit: str = "celsius") -> str:
    return f"Weather in {location}: 25°C, sunny"
```

### 2. Automatic Schema Inference
JSON schemas are automatically generated from Python function signatures:

- Basic types: `str`, `int`, `float`, `bool`
- Collections: `List[T]`, `Dict[K, V]`
- Optional types: `Optional[T]`, `Union[T, None]`
- Enums: Custom enum classes
- Custom classes: Dataclasses and classes with `__annotations__`

### 3. Provider-Specific Tool Formats
Each LLM provider has its own tool format:

- **OpenAI**: Uses `tools` parameter with function definitions
- **Gemini**: Uses `function_declarations` in tools array
- **Anthropic**: Uses `input_schema` format

### 4. Unified Function Registry
All functions are stored in a centralized registry that can convert
to any provider format on demand.

## Architecture

```
LLM (Base Class)
├── FunctionRegistry
│   ├── FunctionDefinition
│   ├── Schema Inference
│   └── Provider Format Conversion
├── OpenAILLM
│   ├── OpenAI Tools Format
│   └── Function Call Handling
└── GeminiLiveModel
    ├── Gemini Tools Format
    └── Real-time Function Calls
```

## Implementation Details

### Function Registry

The `FunctionRegistry` class manages all registered functions:

```python
class FunctionRegistry:
    def __init__(self):
        self._functions: Dict[str, FunctionDefinition] = {}
    
    def function(self, description: str = "", name: Optional[str] = None):
        """Decorator to register a function."""
        def decorator(func: Callable) -> Callable:
            # Infer schema from function signature
            parameters = self._infer_schema(func)
            # Store function definition
            self._functions[func_name] = FunctionDefinition(...)
            return func
        return decorator
```

### Schema Inference

The system analyzes function signatures to generate JSON schemas:

```python
def _infer_schema(self, func: Callable) -> Dict[str, Any]:
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, param.annotation)
        param_schema = self._type_to_json_schema(param_type)
        properties[param_name] = param_schema
        
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }
```

### Adding New Providers
1. Inherit from `LLM` or `RealtimeLLM`
2. Implement `generate_with_functions` method
3. Add provider-specific tool format conversion
4. Handle function call responses

## Usage Examples

### Basic Function Registration
```python
from stream_agents.llm import OpenAILLM

llm = OpenAILLM(name="gpt-4o")

@llm.function("Get weather information")
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 25°C, sunny"

@llm.function("Send notification")
def send_notification(message: str, urgent: bool = False) -> str:
    """Send a notification message."""
    prefix = "🚨 URGENT: " if urgent else "📢 "
    print(f"{prefix}{message}")
    return f"Notification sent: {message}"
```

### Custom Agent with Functions
```python
from stream_agents import Agent

class TaskAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = []
        self._register_functions()
    
    def _register_functions(self):
        @self.llm.function("Add a new task")
        def add_task(task: str, priority: str = "medium") -> str:
            """Add a new task to the list."""
            task_id = len(self.tasks) + 1
            self.tasks.append({
                "id": task_id,
                "task": task,
                "priority": priority,
                "completed": False
            })
            return f"✅ Added task: '{task}' with {priority} priority"
```

### Complex Type Support
```python
from typing import List, Dict, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    HIGH = "high"

@llm.function("Process user data")
def process_users(users: List[Dict[str, str]], priority: Priority = Priority.LOW) -> str:
    """Process a list of users with specified priority."""
    return f"Processed {len(users)} users with {priority.value} priority"
```