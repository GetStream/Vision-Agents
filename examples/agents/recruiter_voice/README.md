# Recruiter (outbound)

Rings a candidate and screens them. Set `OUTBOUND_FROM` (one of your numbers) and
`OUTBOUND_TO` (the handset to ring).

```bash
cd examples/agents/recruiter_voice
uv sync
uv run recruiter_voice.py
```

Needs a router with a telephony vendor and a bought number: see `acceleration/README.md`.
