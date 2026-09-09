# Analyst (text, with a VM)

A text agent whose subagent may run code. `vm=Daytona` on the agent config is the whole of
it: the sandbox belongs to the agent rather than to one conversation, so every session
created from this config has it and no question has to ask for one.

```bash
cd examples/text_agents/analyst
uv sync
uv run analyst.py
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`, plus the Stream app's `STREAM_API_KEY` and
`STREAM_API_SECRET` that every agent is built with. Running the code is Daytona, so the
router needs `DAYTONA_API_KEY`. Daytona is the only sandbox there is today; anything else is refused
when the config is written rather than once a session is running.

Only the subagent is offered the sandbox, because booting one and running code in it takes
seconds and the model holding the conversation has none to spare. So the config names a
subagent as well, and the arithmetic comes back as delegated work.
