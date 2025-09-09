# Function Calling

Function calling allows LLMs to call predefined functions during conversations, enabling them to interact with external systems, perform calculations, and access real-time data.

## Overview

The function calling system consists of:

1. **FunctionRegistry**: Manages available functions and their schemas
2. **LLM Integration**: Base LLM class with function calling support
3. **Provider-specific implementations**: OpenAI, Claude, and Gemini support

## Basic Usage

### Registering Functions

Functions are registered using the `@llm.register_function()` decorator:

```python
from stream_agents.core.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4", api_key="your-api-key")

@llm.register_function(description="Get current weather for a location")
def get_weather(location: str) -> dict:
    """Get the current weather for a location."""
    # Your weather API call here
    return {"location": location, "temperature": "22°C", "condition": "Sunny"}

@llm.register_function(description="Calculate the sum of two numbers")
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
```

### Making Requests

Once functions are registered, the LLM can automatically call them based on user queries:

```python
# The LLM will automatically call get_weather("New York") when asked about weather
response = await llm.simple_response("What's the weather like in New York?")
print(response.text)
```

### Function Parameters

Functions can have:
- **Required parameters**: Must be provided when calling
- **Optional parameters**: Can have default values
- **Type hints**: Used to generate proper JSON schemas

```python
@llm.register_function()
def greet_user(name: str, greeting: str = "Hello", formal: bool = False) -> str:
    """Greet a user with optional formality."""
    prefix = "Good day" if formal else greeting
    return f"{prefix}, {name}!"
```

## Advanced Features

### Custom Function Names

You can specify custom names for functions:

```python
@llm.register_function(name="weather_lookup", description="Get weather data")
def get_weather(location: str) -> dict:
    # Function implementation
    pass
```

### Function Schemas

The system automatically generates JSON schemas for function parameters:

```python
# Get available function schemas
schemas = llm.get_available_functions()
for schema in schemas:
    print(f"Function: {schema['name']}")
    print(f"Description: {schema['description']}")
    print(f"Parameters: {schema['parameters_schema']}")
```

### Manual Function Calls

You can also call functions manually:

```python
# Call a function directly
result = llm.call_function("get_weather", {"location": "London"})
print(result)
```

## Supported LLM Providers

### OpenAI

```python
from stream_agents.core.llm import OpenAILLM

llm = OpenAILLM(model="gpt-4", api_key="your-api-key")
# Functions are automatically included in API calls
```

### Claude (Coming Soon)

```python
from stream_agents.core.llm import ClaudeLLM

llm = ClaudeLLM(model="claude-3-sonnet", api_key="your-api-key")
# Function calling support will be added
```

### Gemini (Coming Soon)

```python
from stream_agents.core.llm import GeminiLLM

llm = GeminiLLM(model="gemini-pro", api_key="your-api-key")
# Function calling support will be added
```

## Error Handling

The system handles function call errors gracefully:

- **Missing functions**: Raises `KeyError` if function not registered
- **Invalid parameters**: Raises `TypeError` for parameter mismatches
- **Runtime errors**: Captured and returned as error results

```python
# Error handling example
try:
    result = llm.call_function("nonexistent_function", {})
except KeyError as e:
    print(f"Function not found: {e}")
```

## Best Practices

1. **Clear descriptions**: Provide meaningful descriptions for functions
2. **Type hints**: Use proper type hints for better schema generation
3. **Error handling**: Implement proper error handling in your functions
4. **Documentation**: Document function parameters and return values
5. **Testing**: Test functions independently before registering

## Example: Complete Function Calling Agent

```python
import asyncio
from stream_agents.core.llm import OpenAILLM

async def main():
    llm = OpenAILLM(model="gpt-4", api_key="your-api-key")
    
    # Register multiple functions
    @llm.register_function(description="Get user information")
    def get_user(user_id: str) -> dict:
        return {"id": user_id, "name": f"User {user_id}", "status": "active"}
    
    @llm.register_function(description="Calculate mathematical expressions")
    def calculate(expression: str) -> float:
        return eval(expression)  # Note: Use safe evaluation in production
    
    # Chat with function calling
    queries = [
        "Get information about user '12345'",
        "Calculate 15 * 7 + 3",
        "What's the status of user '67890' and calculate 100 / 4?"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        response = await llm.simple_response(query)
        print(f"Response: {response.text}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **Functions not being called**: Ensure functions are registered before making requests
2. **Parameter errors**: Check that function parameters match the expected types
3. **API errors**: Verify your API keys and model availability
4. **Schema issues**: Ensure type hints are properly defined

### Debug Mode

Enable debug logging to see function calls:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your function calling code here
```

This will show detailed information about function registration and calls.
