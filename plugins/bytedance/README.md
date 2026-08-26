# Vision Agents ByteDance / BytePlus plugin

Seed Speech STT, TTS and Live Interpretation for [Vision Agents](https://visionagents.ai/).

Wraps the ByteDance / BytePlus (Volcengine) Seed Speech WebSocket APIs:

- **`bytedance.STT`** — streaming ASR (Seed ASR 2.0, `bigmodel_async`)
- **`bytedance.TTS`** — bidirectional streaming TTS (`seed-tts-2.0`)
- **`bytedance.Realtime`** — Live Interpretation (AST 2.0), speech-to-speech translation

## Installation

```bash
uv add "vision-agents[bytedance]"
# or directly
uv add vision-agents-plugins-bytedance
```

## Credentials

Set a new-console API key (recommended):

```bash
export BYTEDANCE_API_KEY="your-api-key"   # BYTEPLUS_API_KEY also works
```

Or a legacy app/access key pair:

```bash
export BYTEDANCE_APP_KEY="your-app-key"
export BYTEDANCE_ACCESS_KEY="your-access-key"
```

## BytePlus regional hosts

The defaults target the mainland Volcengine host
(`wss://openspeech.bytedance.com`). BytePlus accounts are served from a regional
host, and the resource id selecting your SKU can differ too — check your BytePlus
console, then override `ws_url` (and `resource_id` where it differs):

```python
HOST = "wss://voice.ap-southeast-1.bytepluses.com"  # your region's host

stt = bytedance.STT(ws_url=f"{HOST}/api/v3/sauc/bigmodel_async")
tts = bytedance.TTS(ws_url=f"{HOST}/api/v3/tts/bidirection")
realtime = bytedance.Realtime(
    source_language="en",
    target_language="zh",
    ws_url=f"{HOST}/api/v4/ast/v2/translate",
    resource_id="volc.service_type.1000025",
)
```

The values above are the ones `example/` uses; substitute your own region and
resource ids.

## STT + TTS pipeline

```python
from vision_agents.core import Agent
from vision_agents.plugins import bytedance, gemini, getstream

agent = Agent(
    edge=getstream.Edge(),
    stt=bytedance.STT(),
    llm=gemini.LLM(),
    tts=bytedance.TTS(speaker="zh_female_vv_uranus_bigtts"),
)
```

## Live Interpretation

```python
from vision_agents.core import Agent
from vision_agents.plugins import bytedance, getstream

agent = Agent(
    edge=getstream.Edge(),
    llm=bytedance.Realtime(source_language="zh", target_language="en"),
)
```

`bytedance.Realtime` is a translator, not a chat model: it takes speech in the
source language and emits translated speech plus source/translation subtitles.
Text prompts and function calling are not supported.

`mode` selects the output: `"s2s"` (default) returns translated speech and
subtitles, `"s2t"` returns subtitles only. At least one of `source_language` /
`target_language` must be `zh` or `en`.
