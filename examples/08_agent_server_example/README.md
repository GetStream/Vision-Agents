# Agent Server Example

This example shows you how to run AI agents via an HTTP server using [Vision Agents](https://visionagents.ai/).

The `Runner` class provides two modes:

- a single-agent console mode
- and an HTTP server mode that spawns agents on demand.

In this example, we will cover the HTTP server mode which allows you to:

- Spawn agents dynamically via HTTP API
- Manage agent sessions (start, stop, view status)
- Health and readiness checks for load balancers
- Real-time session metrics
- Configurable CORS, authentication, and permissions

## Prerequisites

- Python 3.10 or higher
- API keys for:
    - [Gemini](https://ai.google.dev/) (for the LLM)
    - [Elevenlabs](https://elevenlabs.io/) (for text-to-speech)
    - [Deepgram](https://deepgram.com/) (for speech-to-text)
    - [Stream](https://getstream.io/) (for video/audio infrastructure)

## Installation

1. Go to the example's directory
    ```bash
    cd examples/08_agent_server_example
    ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Create a `.env` file with your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_key
   ELEVENLABS_API_KEY=your_11labs_key
   DEEPGRAM_API_KEY=your_deepgram_key
   STREAM_API_KEY=your_stream_key
   STREAM_API_SECRET=your_stream_secret
   ```

## Running Agent HTTP Server

### Creating the Agent

The `create_agent` function defines how agents are configured:

```python
async def create_agent(**kwargs) -> Agent:
    llm = gemini.LLM("gemini-2.5-flash-lite")

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="My happy AI friend", id="agent"),
        instructions="You're a voice AI assistant...",
        llm=llm,
        tts=elevenlabs.TTS(),
        stt=deepgram.STT(eager_turn_detection=True),
    )

    @llm.register_function(description="Get current weather for a location")
    async def get_weather(location: str) -> Dict[str, Any]:
        return await get_weather_by_location(location)

    return agent
```

### Joining a Call

The `join_call` function handles what happens when an agent joins:

```python
async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("tell me something interesting")
        await agent.finish()
```

### Running with Runner

The `Runner` class ties everything together:

```python
if __name__ == "__main__":
    Runner(
        AgentLauncher(create_agent=create_agent, join_call=join_call),
    ).cli()
```

## Configuration

Customize the HTTP server behavior with `ServeOptions`:

```python
from vision_agents.core import Runner, AgentLauncher, ServeOptions
from fastapi import Depends, HTTPException, Header


# Custom authentication
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")


# Custom permission check — call_id comes from the URL path
async def can_start(call_id: str, x_api_key: str = Header(...)):
    await verify_api_key(x_api_key)


runner = Runner(
    AgentLauncher(create_agent=create_agent, join_call=join_call),
    serve_options=ServeOptions(
        # CORS settings
        cors_allow_origins=["https://myapp.com"],
        cors_allow_methods=["GET", "POST", "DELETE"],
        cors_allow_headers=["*"],
        cors_allow_credentials=True,

        # Permission callbacks (can use FastAPI Depends)
        can_start_session=can_start,
        can_close_session=can_start,
        can_view_session=can_start,
        can_view_metrics=can_start,
    ),
)
```

**Available options:**

| Option                   | Default   | Description                                       |
|--------------------------|-----------|---------------------------------------------------|
| `fast_api`               | none      | Custom FastAPI instance (skips all configuration) |
| `cors_allow_origins`     | `("*",)`  | Allowed CORS origins                              |
| `cors_allow_methods`     | `("*",)`  | Allowed CORS methods                              |
| `cors_allow_headers`     | `("*",)`  | Allowed CORS headers                              |
| `cors_allow_credentials` | `True`    | Allow CORS credentials                            |
| `can_start_session`      | allow all | Permission check for starting sessions. Receives `call_id` from URL path. |
| `can_close_session`      | allow all | Permission check for closing sessions. Receives `call_id` from URL path.  |
| `can_view_session`       | allow all | Permission check for viewing sessions. Receives `call_id` from URL path.  |
| `can_view_metrics`       | allow all | Permission check for viewing metrics. Receives `call_id` from URL path.   |

### Permission Callbacks & Authentication

The `can_start_session`, `can_close_session`, `can_view_session`, and `can_view_metrics` callbacks
are **standard FastAPI dependencies**.

This means they have access to the full power of FastAPI's dependency injection
system:

- **Access request data**: Headers, query parameters, cookies, request body
- **Use `Depends()`**: Chain other dependencies, including database sessions, auth services, etc.
- **Async support**: All callbacks can be `async` functions
- **Automatic validation**: Use Pydantic models for type-safe parameter extraction
- **Raise HTTP exceptions**: Return `401`, `403`, or any status code to deny access

**Example: JWT Authentication**

```python
from fastapi import Depends, Header, HTTPException
from myapp.auth import decode_jwt


async def verify_token(authorization: str = Header(...)):
    """Verify a JWT token from the Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token = authorization.split(" ")[1]
    payload = decode_jwt(token)  # Raises if invalid
    return payload


async def can_start_session(call_id: str, token_payload=Depends(verify_token)):
    """Check if the caller has permission to start agent sessions."""
    if "agents:start" not in token_payload.get("permissions", []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
```

### API Reference

#### OpenAPI & Swagger support

The underlying API is built with `FastAPI` which provides a Swagger UI on http://127.0.0.1:8000/docs.

#### Start a Session

**POST** `/calls/{call_id}/sessions`

Start a new agent and have it join a call.

```bash
curl -X POST http://localhost:8000/calls/my-call-123/sessions \
  -H "Content-Type: application/json" \
  -d '{"call_type": "default"}'
```

**Response:**

```json
{
  "session_id": "agent-uuid",
  "call_id": "my-call-123",
  "session_started_at": "2024-01-15T10:30:00Z"
}
```

#### Close a Session

**DELETE** `/calls/{call_id}/sessions/{session_id}`

Stop an agent and remove it from a call.

```bash
curl -X DELETE http://localhost:8000/calls/my-call-123/sessions/agent-uuid
```

#### Close via sendBeacon

**POST** `/calls/{call_id}/sessions/{session_id}/close`

Alternative endpoint for closing sessions via browser's `sendBeacon()` API.

#### Get Session Info

**GET** `/calls/{call_id}/sessions/{session_id}`

Get information about a running agent session.

```bash
curl http://localhost:8000/calls/my-call-123/sessions/agent-uuid
```

#### Get Session Metrics

**GET** `/calls/{call_id}/sessions/{session_id}/metrics`

Get real-time metrics for a running session.

```bash
curl http://localhost:8000/calls/my-call-123/sessions/agent-uuid/metrics
```

**Response:**

```json
{
  "session_id": "agent-uuid",
  "call_id": "my-call-123",
  "session_started_at": "2024-01-15T10:30:00Z",
  "metrics_generated_at": "2024-01-15T10:35:00Z",
  "metrics": {
    "llm_latency_ms__avg": 245.5,
    "llm_input_tokens__total": 1500,
    "llm_output_tokens__total": 800,
    "stt_latency_ms__avg": 120.3,
    "tts_latency_ms__avg": 85.2
  }
}
```

#### Health Check

**GET** `/health`

Check if the server is alive. Returns `200 OK`.

#### Readiness Check

**GET** `/ready`

Check if the server is ready to spawn new agents. Returns `200 OK` when ready, `400` otherwise.

## Learn More

- [Building a Voice AI app](https://visionagents.ai/introduction/voice-agents)
- [Building a Video AI app](https://visionagents.ai/introduction/video-agents)
- [Simple Agent Example](../01_simple_agent_example) - Basic agent setup
- [Prometheus Metrics Example](../06_prometheus_metrics_example) - Export metrics to Prometheus
- [Deploy Example](../07_deploy_example) - Deploy to Kubernetes
- [Main Vision Agents README](../../README.md)
