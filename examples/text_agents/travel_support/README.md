# Travel support chat agent

A customer support chatbot for a travel agency, set up from this directory and tested in the
dashboard's chat. [`../travel_support_js`](../travel_support_js) is the same agent in Node.

| File                         | What it is                                                     |
| ---------------------------- | -------------------------------------------------------------- |
| `agent.yaml`                 | A text agent with the TableJourney connector and its tools     |
| `instructions.md`            | Who the agent is and how it answers                            |
| `knowledge/travel_policy.md` | Bookings, changes, refunds, baggage, insurance and contact     |
| `skills/itinerary.md`        | How it plans a day-by-day trip, and the markdown it answers in |

[TableJourney](https://tablejourney.com/agents/) is a public MCP server with verified
restaurants, sights, tours and stays. It is not in the router's catalog, so the script creates
it as a custom connector, `custom_tablejourney`, with an app connection labelled
`tablejourney`. Both are created once; later runs leave them as they are. `agent.yaml` offers
only its read-only tools.

`lookup_booking` in `travel_support.py` is a function the agent calls, run by this process.
After syncing, the script keeps running as a worker that hosts it (`stream.Dispatch().host`),
so the router offers it to every conversation with the agent, the ones the dashboard opens
included, and sends each call here.

## Run it

You need a router (see [acceleration/README.md](../../../acceleration/README.md)) and the
repo-root `.env` with `STREAM_ACCELERATION_URL`, `STREAM_API_KEY`, `STREAM_API_SECRET` and
`DASHBOARD_BASE_URL`, the app's agents page on the dashboard. A local router also needs
`ROUTER_CONNECTORS_ENABLED=true`, or it refuses the connector.

```bash
uv sync
uv run travel_support.py
```

The script syncs the agent, opens its test page on the dashboard and hosts `lookup_booking`
until you stop it. Try:

- "How much does it cost to change my flight?"
- "Which flight am I on? My booking is WF7K2Q."
- "Plan 2 days in Lisbon for a food lover."
