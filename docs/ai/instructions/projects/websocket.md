
# Design & Implementation Guide
Streaming Py – WebSocket client for the Video Coordinator (`StreamAPIWS`)

---

## 1  Overview
`StreamAPIWS` is an asynchronous WebSocket client that lets the SDK talk to Stream’s Video Coordinator at
`wss://chat.stream-io-api.com/api/v2/connect?api_key=<API_KEY>`.

Key capabilities

• Authenticate immediately after the TCP/WS handshake
• Emit all server events through a `pyee.AsyncIOEventEmitter` interface (`event.type` is the channel)
• Maintain liveness with heart-beats (send every 25 s, expect any inbound message within 30 s)
• Recover transparently from network drops using an exponential-back-off retry loop (max 5 attempts)
• Expose an explicit `disconnect()` coroutine for graceful shutdown
• Follow existing logging conventions (`logger = logging.getLogger(__name__)`, no global config)

The public surface is intentionally small:

```python
ws_client = StreamAPIWS(api_key, token, user_id)
connected_payload = await ws_client.connect()  # raises if first msg.type == "error"

@ws_client.on("health.check")
async def _(_: dict): ...
...
await ws_client.disconnect()
```

---

## 2  Dependencies & Versions

| Purpose             | Package        | Min-version | Notes |
|---------------------|----------------|-------------|-------|
| Async WebSockets    | websockets     | 12 .x       | Pure-Python, CPython-only OK |
| Event dispatcher    | pyee           | 13 .x       | Already optional in `webrtc` group |
| Async HTTP helper   | aiohttp        | 3.11 .x     | Already in `pyproject.toml` (used only for JWT fetch tests) |
| Testing             | pytest-asyncio | ≥0.23.8     | In dev-deps |

Add `websockets>=12,<13` to the existing `webrtc` optional-dependency group and run `uv add websockets --group webrtc`.

Supported Python: 3.9 – 3.12 (per project `requires-python`).

---

## 3  Directory & Module Layout

```
getstream/
└─ video/
   └─ rtc/
      └─ coordinator/
         ├─ __init__.py          # Public re-exports
         ├─ ws.py                # StreamAPIWS implementation
         ├─ backoff.py           # Reusable exponential strategy (tiny helper)
         └─ tests/
            ├─ test_connect.py
            ├─ test_heartbeat.py
            ├─ test_reconnect.py
            └─ assets/           # JSON fixtures if needed
```

---

## 4  API Shape

```python
class StreamAPIWS(pyee.AsyncIOEventEmitter):
    def __init__(
        self,
        api_key: str,
        token: str,
        user_id: str,
        *,
        uri: str = DEFAULT_WS_URI,
        healthcheck_interval: float = 25.0,
        healthcheck_timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_factor: float = 2.0,
        logger: logging.Logger | None = None,
    ): ...

    async def connect(self) -> dict:
        """Open the socket, authenticate, wait for first server frame.
        Returns the first event payload (usually {"type": "connection.ok"}).
        Raises StreamWSConnectionError if the first frame has type == "error"
        or if authentication times out.
        """

    async def disconnect(self) -> None:
        """Cancel internal tasks and close the WebSocket cleanly."""
```

All kwargs have sensible defaults but remain configurable for unit tests.

---

## 5  Connection Lifecycle

1. **TCP / TLS / WS handshake** via `websockets.connect(uri, ..., ping_interval=None)`
   (disable the library’s automatic pings; we handle heart-beats ourselves).

2. **Send authentication payload immediately**

```json
{
  "token": "<jwt_token>",
  "products": ["video"],
  "user_details": {"id": "<user_id>"}
}
```

3. **Await the first server message**
   • If `payload["type"] == "error"` → raise `StreamWSAuthError(payload)` (no retry).
   • Otherwise emit `payload` (`event.type` channel) and return it to caller.

4. **Start two background tasks**
   a. **_reader_task** – forever `await ws.recv()`, JSON-parse, emit.
      • Reset `last_received` timestamp on every frame.
      • On `websockets.exceptions.ConnectionClosedError`, jump to reconnection.
   b. **_heartbeat_task** – every `healthcheck_interval` seconds:
      • Send `{"type":"health.check"}`.
      • If `now - last_received > healthcheck_timeout`, treat as dead connection → reconnection.

5. **Reconnection strategy**
   • `attempt = 0`
   • For network-driven closures OR heartbeat timeouts:
     ```
     while attempt < max_retries:
         delay = backoff_base * backoff_factor**attempt   # 1,2,4,8,16
         await asyncio.sleep(delay)
         try: await _open_socket_again()
         except OSError: attempt += 1; continue
         else: break
     else:
         raise StreamWSMaxRetriesExceeded()
     ```
   • On success, authentication & first-message steps repeat transparently; internal tasks restart; events continue flowing.

6. **Graceful shutdown** (`disconnect()`):
   • Cancel heartbeat & reader tasks (with `asyncio.CancelledError` swallowed).
   • Close the WS (`await ws.close(code=1000, reason="client disconnect")`).

---

## 6  Event Dispatching

`pyee.AsyncIOEventEmitter` provides `.on()`, `.once()`, `.emit()`.
Implementation detail: internally call `super().emit(event_type, payload)` on every inbound frame.
Handlers receive the raw JSON dict (already parsed).

---

## 7  Logging

Use module-level logger:

```python
logger = logging.getLogger(__name__)

logger.info("Connecting to coordinator", extra={"uri": uri})
logger.debug("Sending auth payload")
logger.warning("Reconnect attempt %s/%s failed", attempt, max_retries, exc_info=err)
```

No handler configuration; rely on application to wire sinks & levels.

---

## 8  Exceptions

Create small hierarchy in `getstream.video.rtc.coordinator.errors`:

```python
class StreamWSException(Exception): ...
class StreamWSAuthError(StreamWSException): ...
class StreamWSMaxRetriesExceeded(StreamWSException): ...
class StreamWSConnectionError(StreamWSException): ...
```

First-frame `"error"` -> `StreamWSAuthError`, transient socket failures -> `StreamWSConnectionError`.

---

## 9  Testing Strategy

• **Happy path**: use `pytest-asyncio` with a local `websockets.server` that echoes health checks.
• **Error-on-first‐frame**: server yields `{"type":"error"}` then closes; ensure `connect()` raises & no retries.
• **Heartbeat timeout**: freeze time or delay server responses; verify client reconnects after 30 s.
• **Exponential back-off**: monkey-patch `asyncio.sleep` to record delays (1,2,4,8,16).
• **Disconnect API**: mock ws.close(), ensure tasks are cancelled.

Helper fixtures can live in `tests/conftest.py`.

---

## 10  Future Extensions (out-of-scope for v1)

• Custom back-off strategies via callable injection
• Jitter addition to back-off
• Multi-consumer fan-out (`async for payload in ws_client.listen("*")`)
• TLS termination metrics, RTT histogram export

---

## 11  Next Steps

1. Add `websockets` to the `webrtc` dep group (`uv add websockets --group webrtc`).
2. Scaffold modules & tests as per §3.
3. Implement `backoff.py`, then `ws.py` iteratively with unit tests.
4. Run `uv run pytest` and `uv run ruff format getstream/ tests/`.
5. Write library changelog & update README.

Once you’re happy with this design, we can start coding the implementation.



Below is a Cursor-friendly “recipe” that breaks the whole feature into 7 incremental, self-contained steps.
Follow them in order; after finishing each one, run the indicated commands to verify that everything still passes before moving on.

-------------------------------------------------
STEP 1 – Add the external dependency
-------------------------------------------------
Files to touch
• `pyproject.toml`

Actions
1. Under `[project.optional-dependencies].webrtc` append
   ```
   "websockets>=12,<13",
   ```
2. From the project root, install the new dep:
   ```
   uv add websockets --group webrtc
   uv sync --dev --all-packages
   ```
Tests
• Nothing to run yet, but confirm lockfile updates cleanly:
```
uv run python -c "import websockets, pyee, aiohttp; print('deps OK')"
```

-------------------------------------------------
STEP 2 – Scaffold the module layout
-------------------------------------------------
Files/directories to create
• `getstream/video/rtc/coordinator/__init__.py`
• `getstream/video/rtc/coordinator/ws.py` (empty placeholder)
• `getstream/video/rtc/coordinator/backoff.py` (empty placeholder)
• `getstream/video/rtc/coordinator/errors.py` (empty placeholder)

Actions
1. Create each file with just a docstring and `logger = logging.getLogger(__name__)` (where relevant).
2. Re-export the public class in `__init__.py`:
   ```python
   from .ws import StreamAPIWS  # noqa: F401
   ```
Tests
• Run the full suite to ensure you didn’t break import graphs:
```
uv run pytest -q
```

-------------------------------------------------
STEP 3 – Implement exponential-back-off helper
-------------------------------------------------
File to edit
• `getstream/video/rtc/coordinator/backoff.py`

Actions
1. Add `async def exp_backoff(max_retries:int, base:float=1.0, factor:float=2.0): -> AsyncIterator[float]`
2. Yield the successive delays (e.g. 1,2,4,8,16) until `max_retries` exhausted.

Tests
Create `getstream/video/rtc/coordinator/tests/test_backoff.py` with a small `pytest-asyncio` test that collects the yielded seconds and asserts the sequence equals `[1,2,4,8,16]` for `max_retries=5`.

Run
```
uv run pytest getstream/video/rtc/coordinator/tests/test_backoff.py -q
```

-------------------------------------------------
STEP 4 – Define the error hierarchy
-------------------------------------------------
File to edit
• `getstream/video/rtc/coordinator/errors.py`

Actions
Implement:
```python
class StreamWSException(Exception): ...
class StreamWSAuthError(StreamWSException): ...
class StreamWSConnectionError(StreamWSException): ...
class StreamWSMaxRetriesExceeded(StreamWSException): ...
```

Tests
Add `tests/test_errors.py` verifying simple inheritance relationships.

Run
```
uv run pytest getstream/video/rtc/coordinator/tests/test_errors.py -q
```

-------------------------------------------------
STEP 5 – Build the minimal StreamAPIWS (connect / disconnect only)
-------------------------------------------------
File to edit
• `getstream/video/rtc/coordinator/ws.py`

Actions
1. Make `StreamAPIWS` inherit from `pyee.AsyncIOEventEmitter`.
2. Implement `__init__`, `_build_auth_payload`, `connect()`, `disconnect()`.
   – For now, skip heartbeat and retry logic; simply open the socket, send auth, wait for first frame, raise `StreamWSAuthError` if `type=="error"`.
3. Add module-level logger.

Tests
`tests/test_connect.py`
• Spin up an in-process mock server with `websockets.serve` that sends a single `{"type":"connection.ok"}` frame.
• Assert `connect()` returns that payload.
• Another test where the server sends `{"type":"error"}` and ensure `StreamWSAuthError` is raised and the socket is closed.

Run
```
uv run pytest getstream/video/rtc/coordinator/tests/test_connect.py -q
```

-------------------------------------------------
STEP 6 – Add heartbeat (25 s send, 30 s receive) and auto-reconnect
-------------------------------------------------
File to edit
• `getstream/video/rtc/coordinator/ws.py`

Actions
1. Add two background tasks started by `connect()`:
   • `_reader_task` – reads frames, emits events, updates `self._last_received`.
   • `_heartbeat_task` – every 25 s sends `{"type":"health.check"}`; if `now - _last_received > 30 s` triggers `_reconnect()`.
2. Implement internal `_reconnect()` using the helper from step 3 with `max_retries=5`.

Tests
`tests/test_heartbeat.py`
• Use `freezegun.advance_by` or manual `asyncio.sleep` patching to imitate time passage and verify reconnect is attempted after 30 s of silence.
• Ensure that on success the client continues emitting events.

Run
```
uv run pytest getstream/video/rtc/coordinator/tests/test_heartbeat.py -q
```

-------------------------------------------------
STEP 7 – Final polish: logging, ruff, full suite
-------------------------------------------------
Files to touch
• `getstream/video/rtc/coordinator/ws.py`  (add informative `logger.debug/info/warning` calls)
• Any `__init__.py` for re-exports if needed.

Actions
1. Ensure every catch-block logs `exc_info`.
2. Run formatter & linter:
   ```
   uv run ruff format getstream/ tests/
   uv run ruff check getstream/ tests/ --fix
   ```
3. Run the entire test matrix:
   ```
   uv run pytest -q
   ```

If everything is green, the feature is complete and ready for PR review.

-------------------------------------------------
How to iterate
Work through one step at a time, commit, run the specified tests, and only then proceed.
This keeps the Cursor change-set small and the feedback loop fast.
