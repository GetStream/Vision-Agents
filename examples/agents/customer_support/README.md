# Customer support

Joins a call as the agent `agent.yaml` names, storing this directory on the acceleration
server on the way in.

```bash
cd examples/agents/customer_support
uv sync
uv run customer_support.py
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`.
