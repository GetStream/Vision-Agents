# Describe an image

Send an image URL through `stream.LLM(target="vlm").responses.create(...)` and print the streamed description. Use `ImageContent(data=..., mime="image/png")` for local bytes. Multiple images retain their order. Text-only providers reject image requests.

Run against a local router, with `STREAM_ACCELERATION_URL` and `STREAM_ACCELERATION_CUSTOMER_ID` set in the repo-root `.env`:

```bash
uv sync
uv run describe_image.py
```

`vlm` selects an image-capable provider before the socket opens. Override it with `STREAM_ACCELERATION_LLM=provider/model`. `llm-fast` keeps its existing selection policy.
