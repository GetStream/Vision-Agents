# Docs agent (text)

A text agent that answers out of two things it has read: the `knowledge/` directory next to
this file, and a page on the docs site.

```bash
cd examples/text_agents/docs_agent
uv sync
uv run docs_agent.py
```

Needs a router: see `acceleration/README.md`, then `STREAM_ACCELERATION_URL` and
`STREAM_ACCELERATION_CUSTOMER_ID`. Both halves of what it knows are the router's: the
knowledge base wants `TURBOPUFFER_API_KEY` and reading a page wants `EXA_API_KEY`. Without
them the router says so rather than answering out of nothing.

Both end up in one namespace, named after this directory, so a single lookup mid-answer
covers both. A page is a subscription rather than a one-off: adding it again re-reads it and
replaces the passages it wrote last time.
