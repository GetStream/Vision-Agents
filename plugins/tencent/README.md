# Vision Agents Plugin: Tencent TRTC Edge

Tencent TRTC (Real-Time Communication) edge transport for Vision Agents. Lets an agent join a Tencent TRTC room and exchange audio/video with participants using the [Tencent LiteAV SDK](https://cloud.tencent.com/document/product/647) Python bindings.

## Requirements

- **Linux**, x86_64 or aarch64. The underlying [`liteav`](https://pypi.org/project/liteav/) package ships only manylinux wheels; macOS and Windows are not supported natively. See [Running on macOS](#running-on-macos) below.
- **User sig**: You must supply a valid `user_sig` for room entry (e.g. generate with [TLSSigAPIv2](https://cloud.tencent.com/document/product/647/34399) or your backend). Alternatively pass `key=` and the plugin will sign per join via the bundled `tls-sig-api-v2` helper.

## Install

```bash
uv add vision-agents-plugins-tencent
```

On Linux, this pulls `liteav` from PyPI. On macOS the `liteav` dependency is skipped via a platform marker, so the package installs but `tencent.Edge()` raises at runtime.

## Usage

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import tencent

agent_user = User(id="agent-1", name="Agent")
edge = tencent.Edge(
    sdk_app_id=YOUR_SDK_APP_ID,
    user_sig=USER_SIG,  # or key=KEY to generate per join
)

agent = Agent(
    edge=edge,
    agent_user=agent_user,
    llm=...,
    # stt, tts, etc.
)

call = await edge.create_call(call_id=str(ROOM_ID), room_id=ROOM_ID)
await agent.join(call)
await agent.finish()
```

## Running on macOS

The Tencent LiteAV SDK ships as a Linux shared library and cannot be loaded natively on macOS. Run the example inside a Linux Docker container:

```bash
cd plugins/tencent
docker compose build
docker compose run --rm tencent-agent
```

The Call ID is logged on join so you can connect from your preferred client.

## Testing the example end-to-end

The bundled `example/tencent_edge_example.py` joins a fixed room (`12345` by default, override with `TENCENT_TEST_ROOM_ID`) and waits indefinitely for a participant. A minimal browser test client lives at `example/test_client.html` and is served by the `test-client` docker-compose service.

1. **Start the agent**:

   ```bash
   cd plugins/tencent
   docker compose run --rm tencent-agent
   ```

   The `test-client` service starts automatically as a dependency and serves `example/` on `http://localhost:8000/`. It stays up between agent runs; stop it explicitly with `docker compose down` when done.

   On join the agent logs a clickable URL like:

   ```
   🔗 Open the test client and click Join:
       http://localhost:8000/test_client.html?appid=<your-sdk-app-id>&user=test-user-1&room=12345&sig=...
   ```

   The URL has `sdk_app_id`, room, user, and a freshly generated `user_sig` baked in, so the HTML form auto-fills. Browsers require a user gesture for `getUserMedia`, so we can't auto-submit.

2. **Click the URL, then Join in the browser**. Do this before the agent finishes greeting — ElevenLabs (and most realtime STT providers) drop their websocket after ~15 s of silence, so the agent needs a participant on join to receive audio immediately. The example calls `agent.simple_response(...)` right after join, so you should hear the agent's greeting within a couple of seconds of clicking Join.

3. **Speak**. The flow is `Tencent → STT → LLM → TTS → Tencent → browser`. Watch the agent log for `🎤 [Transcript Complete]: …` (STT), `🤖 [LLM response final]: …` (LLM), and absent `TencentAudioTrack` write errors (TTS → outgoing track).

The HTML form also persists fields in `localStorage`, so reloading without URL params still keeps the last values.

### Known noise

- `AudioQueue buffer limit exceeded: dropped 1 chunks (20.0ms)` — fires during long TTS replies when mic echo (browser ↔ agent) backs up the incoming queue. The framework drops oldest 20 ms chunks to keep latency bounded. Harmless; not specific to Tencent.
- `ElevenLabs WebSocket connection closed` after idle — STT server-side idle disconnect. The plugin reconnects on the next error but not on a clean close (separate `vision-agents-plugins-elevenlabs` issue).

## Configuration

- **sdk_app_id** (int): Tencent TRTC SDK App ID. Falls back to `TENCENT_SDKAppID` env var.
- **user_sig** (str): User signature for the agent user.
- **key** (str): Optional. If set and `user_sig` is not, the plugin generates `user_sig` via TLSSigAPIv2. Falls back to `TENCENT_SDKSecretKey` env var.
- **video_fps** (int): Outgoing video frame rate.

### Environment variables

- `TENCENT_SDKAppID`, `TENCENT_SDKSecretKey` — credentials.
- `TENCENT_TRTC_SCENE` — one of `auto` (default), `videocall`, `call`, `record`.
