
Go backend. OpenAPI. Ruedis for redis, Bun for Postgres, Goose for migrations. Testify for suites. 
Go code goes in the acceleration folder

Requirements

Verify cursor cloud agents are setup
That you have baseten access and you have Stream credentials in .env
Install https://github.com/basetenlabs/truss and log me in
Verify you have deepgram credentials in .env

Sprint 1:

Step 1.

Run https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 on Baseten

Step 2.

Write a Go based STT abstraction that supports both Parakeet and Deepgram Flux
Study our python codebase for STT to understand what parts of STT needs to be standardized.

Also read our standardization guidelines.

Step 3.

Add an STT router abstraction. We want to track the following

- provider/model

Customer level stats
- Performance
- Uptime
- Audio duration
- Number of API calls
- Costs

Aggregated stats (hourly/daily etc)
- Performance
- Uptime
- Token totals
- Total API calls

Step 4.

In addition to router stats we also want to support some shortcuts like

- en-low-latency
- multilingual-low-latency
- en-high-accuracy
- multilingual-high-accuracy

Step 5.

Connect to a call, and connect the audio to our STT Router.
Show the transcription results in the terminal.

