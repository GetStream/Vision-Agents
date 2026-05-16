# Vision Agents Plugin: Tencent TRTC Edge

Tencent TRTC (Real-Time Communication) edge transport for Vision Agents. Lets an agent join a Tencent TRTC room and exchange audio/video with participants using the [Tencent LiteAV SDK](https://cloud.tencent.com/document/product/647) Python bindings.

## Quickstart

Talk to an agent in 5 minutes. You'll need Docker, an `.env` with `TENCENT_SDKAppID`, `TENCENT_SDKSecretKey`, `GOOGLE_API_KEY`, `CARTESIA_API_KEY`, and `ELEVEN_API_KEY` at the repo root, and a working microphone in a Chromium-based browser.

1. **Open Tencent's hosted TRTC Web SDK quick demo:**
   <https://web.sdk.qcloud.com/trtc/webrtc/v5/demo/quick-demo-js/index.html>

   - Paste your `TENCENT_SDKAppID` into **SDKAppID** and `TENCENT_SDKSecretKey` into **SDKSecretKey** — the page generates `UserSig` client-side.
   - Leave the auto-generated **UserID** and **RoomID(String)** as is.
   - Click **Enter Room** → demo log should print `🟩 [user_***] enterRoom.`
   - Click **Start Local Video** — this also publishes the mic.

2. **Copy the `RoomID(String)` from the demo form** and launch the agent with it:

   ```bash
   cd plugins/tencent
   TENCENT_TEST_ROOM_ID=<paste-room-id-here> docker compose run --rm tencent-agent
   ```

   On first run this builds the image and resolves the workspace; subsequent runs are fast. When the agent joins you'll see `Tencent TRTC OnRemoteUserEnterRoom: <userId>` matching the **UserID** shown in the demo.

3. **Talk.** The flow is `browser → Tencent → STT (ElevenLabs) → LLM (Gemini) → TTS (Cartesia) → Tencent → browser`. Confirm each leg in the agent log:
   - `🎤 [Transcript Complete]: …` — STT got your speech.
   - `🤖 [LLM response final]: …` — LLM produced a reply.
   - Reply plays back in the browser.

## Usage in your own code

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

## Install

```bash
uv add vision-agents-plugins-tencent
```

On Linux this pulls `liteav` from PyPI. On macOS the `liteav` dependency is skipped via a platform marker, so the package installs but `tencent.Edge()` raises at runtime — use the Docker setup from the Quickstart.

## Configuration

`tencent.Edge(...)` parameters:

- **sdk_app_id** (int): Tencent TRTC SDK App ID. Falls back to `TENCENT_SDKAppID` env var.
- **user_sig** (str): User signature for the agent user.
- **key** (str): Optional. If set and `user_sig` is not, the plugin generates `user_sig` via TLSSigAPIv2. Falls back to `TENCENT_SDKSecretKey` env var.
- **video_fps** (int): Outgoing video frame rate.

Environment variables:

- `TENCENT_SDKAppID`, `TENCENT_SDKSecretKey` — credentials.
- `TENCENT_TRTC_SCENE` — one of `auto` (default), `videocall`, `call`, `record`.
- `TENCENT_TEST_ROOM_ID` — used only by `example/tencent_edge_example.py`.

## Platform support

The underlying [`liteav`](https://pypi.org/project/liteav/) package ships only manylinux wheels — **Linux x86_64 / aarch64 only**. macOS and Windows must run the agent inside a Linux container (see Quickstart). User sigs are needed for room entry; either generate them with [TLSSigAPIv2](https://cloud.tencent.com/document/product/647/34399) or pass `key=` and let the plugin sign per join.
