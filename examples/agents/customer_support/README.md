# Customer support

Syncs this directory onto the acceleration server, then joins a call as that agent.

```bash
cd examples/agents/customer_support
uv sync
uv run customer_support.py
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`.
