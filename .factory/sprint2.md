
read overview.md and standardization.md for general guidelines


Sprint 2

Step 1

Spin up https://huggingface.co/fishaudio/s2-pro on baseten

Step 2.

Write a Go based TTS abstraction that supports both Fish and Elevenlabs.
Also support Qwen-Audio-3.0-TTS Flash and Plus (slower)

Also read our standardization guidelines.

Step 3.

Add an TTS router abstraction. We want to track the following

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

Expose a CLI command in go, type the text and play the audio. Thx