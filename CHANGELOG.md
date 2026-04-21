# v0.4.0

## Breaking Changes

### Built-in HTTP server

#### API Endpoints

All session endpoints now include `call_id` as a path parameter:

| Before                               | After                                                |
|--------------------------------------|------------------------------------------------------|
| `POST /sessions`                     | `POST /calls/{call_id}/sessions`                     |
| `DELETE /sessions/{session_id}`      | `DELETE /calls/{call_id}/sessions/{session_id}`      |
| `POST /sessions/{session_id}/close`  | `POST /calls/{call_id}/sessions/{session_id}/close`  |
| `GET /sessions/{session_id}`         | `GET /calls/{call_id}/sessions/{session_id}`         |
| `GET /sessions/{session_id}/metrics` | `GET /calls/{call_id}/sessions/{session_id}/metrics` |

#### Request Body

- `call_id` removed from `POST /sessions` request body — now a URL path parameter

#### Response Codes

- `DELETE /calls/{call_id}/sessions/{session_id}` now returns **202 Accepted** (was 204)
- `POST /calls/{call_id}/sessions/{session_id}/close` now returns **202 Accepted** (was 200)
- Session closure is now asynchronous — the owning node processes the close request on its next maintenance cycle

#### ServeOptions

- `get_current_user` option removed
- Permission callbacks (`can_start_session`, `can_close_session`, `can_view_session`, `can_view_metrics`) now receive `call_id: str` as a parameter

#### Removed Dependencies

- `vision_agents.core.runner.http.dependencies.get_session` removed
- `vision_agents.core.runner.http.dependencies.get_current_user` removed

### FunctionRegistry

- `register_function()` now rejects synchronous functions with `ValueError` — all registered functions must be async (#373)
- `FunctionRegistry.call_function()` and `LLM.call_function()` are now `async` (#373)

### Agent authentication

- `Agent.create_user()` renamed to `Agent.authenticate()` (#380)
- `EdgeTransport.create_user()` renamed to `EdgeTransport.authenticate()` (#380)
- Authentication is now called automatically during `Agent.start()` — manual calls are no longer needed (#380)

### Testing

- `mock_tools()` and standalone `mock_functions()` removed from `vision_agents.testing` (#376)
- Use `TestSession.mock_functions()` instead

### AgentLauncher

- `cleanup_interval` parameter renamed to `maintenance_interval` (#374)
- `created_by` parameter removed from `start_session()` (#374)
- `AgentSession.created_by` field removed (#374)
- `call_id` values must match `^[a-z0-9_-]+$` (raises `InvalidCallId`) (#374)
- New parameter: `registry: SessionRegistry | None = None` (#374)

## New Features

### Inworld Realtime plugin (WebRTC)

Adds `inworld.Realtime` for low-latency speech-to-speech over Inworld's Realtime API (WebRTC transport). Protocol-compatible with OpenAI Realtime — supports function calling, turn detection, and multiple upstream models via the `<provider>/<model>` ID format (e.g. `"openai/gpt-4o-mini"`, `"google-ai-studio/gemini-2.5-flash"`). (#502)

### Redis-backed Agent session registry for horizontal scaling

Sessions are shared across nodes via Redis, enabling cross-node session queries and closure without sticky sessions. (#374)

```python
from vision_agents.core import AgentLauncher, Runner
from vision_agents.core.agents.session_registry import RedisSessionKVStore, SessionRegistry

store = RedisSessionKVStore(url="redis://localhost:6379")
registry = SessionRegistry(store=store)
runner = Runner(AgentLauncher(create_agent=create_agent, join_call=join_call, registry=registry))
```

Install with: `uv add "vision-agents[redis]"`

**New public API on AgentLauncher:**

| Method                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `get_session_info(call_id, session_id)`      | Query session info from shared storage |
| `request_close_session(call_id, session_id)` | Request closure from any node          |

**Custom store backends:** `SessionKVStore` is an abstract class that can be subclassed to support any TTL key-value store (DynamoDB, Memcached, etcd, etc.).

### PEP 561 compliance

`py.typed` markers added to `vision_agents.core` and `vision_agents.testing` for downstream type checking support. (#378)

## Bug Fixes

- **EventManager**: fix crash when event handlers have return type annotations (#381)
- **RedisSessionKVStore**: fix import error when `redis` package is not installed (#384)
- **Agent metrics**: fix metrics storage and serialization in session registry (#387)
