# OpenAI Plugin for GetStream

This package provides OpenAI integration for the GetStream plugin ecosystem.

It enables features such as:
- Real-time transcription and language processing using OpenAI models
- Easy integration with other GetStream plugins and services
- Function calling capabilities for dynamic interactions

## Installation

```bash
pip install getstream-plugins-openai
```

## Usage

```python
from getstream.plugins.openai import OpenAIRealtime

# Initialize with API key
sts = OpenAIRealtime(api_key="your_openai_api_key", voice="alloy")

# Connect to a call
async with await sts.connect(call, agent_user_id="assistant") as connection:
    # Send user message
    await sts.send_user_message("Hello, how can you help me?")

    # Request assistant response
    await sts.request_assistant_response()
```

## Function Calling

The OpenAI Realtime API supports function calling, allowing the assistant to invoke custom functions you define. This enables dynamic interactions like:

- Database queries
- API calls to external services
- File operations
- Custom business logic

### Example with Function Calling

```python
from getstream.plugins.openai import OpenAIRealtime

# Define your functions
def get_weather(location: str) -> str:
    """Get current weather for a location"""
    # Your weather API logic here
    return f"Weather in {location}: Sunny, 72°F"

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    # Your email sending logic here
    return f"Email sent to {to} with subject: {subject}"

# Initialize with functions
sts = OpenAIRealtime(
    api_key="your_openai_api_key",
    voice="alloy",
    functions=[
        {
            "name": "get_weather",
            "description": "Get current weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "send_email",
            "description": "Send an email to someone",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    ]
)

async with await sts.connect(call, agent_user_id="assistant") as connection:
    await sts.send_user_message("What's the weather like in San Francisco?")
    await sts.request_assistant_response()

    # The assistant can now call your functions and you can respond with results
    # await sts.send_function_call_output("call_id", "function_result")
```

## Requirements
- Python 3.10+
- openai[realtime] api
- GetStream SDK

## License
MIT
