# Vision Agents Plugin: Tencent TRTC Edge

Tencent TRTC (Real-Time Communication) edge transport for Vision Agents. Lets an agent join a Tencent TRTC room and exchange audio with participants using the [Tencent LiteAV SDK](https://cloud.tencent.com/document/product/647) Python bindings.

## Requirements

- **LiteAV Python module**: Build and install the `liteav` module from the Tencent RTC SDK (see the SDK's `python/python_x86_64/README` or `build.sh`). The plugin imports `liteav` at runtime; if it is not installed, using `tencent.Edge()` will raise with instructions.
- **User sig**: You must supply a valid `user_sig` for room entry (e.g. generate with [TLSSigAPIv2](https://cloud.tencent.com/document/product/647/34399) or your backend).

## Usage

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import tencent

agent_user = User(id="agent-1", name="Agent")
edge = tencent.Edge(
    sdk_app_id=YOUR_SDK_APP_ID,
    user_sig=USER_SIG,  # or use key=KEY to generate per join
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

## Configuration

- **sdk_app_id** (int): Tencent TRTC SDK App ID.
- **user_sig** (str): User signature for the agent user (recommended).
- **key** (str): Optional. If set and `user_sig` is not, the plugin will try to generate user_sig via TLSSigAPIv2 (requires `TLSSigAPIv2` package).

No changes are made to the Vision Agents core; this is a plugin-only implementation.
