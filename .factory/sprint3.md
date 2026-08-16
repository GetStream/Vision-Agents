
read overview and standardization

time to add an LLM class

Step 1. 

See how to use deepseek flash on baseten. 

Step 2.

Write a Go based LLM abstraction that supports both Deepseek flash, gemma 4 and openAI.

Also read our standardization guidelines.

Step 3.

Add an LLM router abstraction. We want to track the following

- provider/model

Customer level stats
- Performance
- Uptime
- Token usage
- Number of API calls
- Costs

Aggregated stats (hourly/daily etc)
- Performance
- Uptime
- Token totals
- Total API calls

Step 4.

In addition to router stats we also want to support some shortcuts like

- llm-fast

Step 5. Create a Go agent class

Make it similar to the agent definition we use in Vision agents/ python. see the examples folder

Step 6. Add stream support

Verify that we can join a call with Stream's go sdk.
See the docs in workspace/getstream-go-webrtc/

