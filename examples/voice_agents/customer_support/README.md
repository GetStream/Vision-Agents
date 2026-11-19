# Customer support

Stores this directory on the acceleration server with `agent.sync()`, then joins a call as
the agent `agent.yaml` names.

```bash
cd examples/voice_agents/customer_support
uv sync
uv run customer_support.py
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`.
