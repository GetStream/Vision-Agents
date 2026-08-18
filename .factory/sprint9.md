
# Hosting

Add a docker image for running the Go API easily. 

Where do we run these voice Agents? different GCP project?

# Mem0

Implement mem0 if we haven't done it yet on the Go side of things. 

# Dashboard

Create a dashboard folder with a nice React app example that shows.

Overview
- Show the last 5 running calls
- Agent configs
- AI costs by tag

Agent configs CRUD
- Name
- STT
- TTS
- Voice model
- Thinking model
- Skills setup (add skill lookup support to our harness)
- Knowledge/RAG (turbopuffer best practices based)

Phone numbers
- Searching numbers
- Buying a number

Campaign API (for outbound calling)

- Define concurrency
- Define the skill that should be used (point to an agent config)
- Share some custom instructions per user that you're calling

Call detail
- Show the call summary
- Show a visualization of who is talking when similar to Gong
- Show a transcription
- Call review score

