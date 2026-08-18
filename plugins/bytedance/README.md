# Vision Agents ByteDance / BytePlus plugin

Seed Speech STT, TTS and Live Interpretation for [Vision Agents](https://visionagents.ai/).

Wraps the ByteDance / BytePlus (Volcengine) Seed Speech WebSocket APIs:

- **`bytedance.STT`** — streaming ASR (Seed ASR 2.0, `bigmodel_async`)
- **`bytedance.TTS`** — bidirectional streaming TTS (`seed-tts-2.0`)
- **`bytedance.Realtime`** — Live Interpretation (AST 2.0), speech-to-speech translation

## Installation

```bash
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

## STT + TTS pipeline

```python
from vision_agents.core import Agent
from vision_agents.plugins import bytedance, openai, getstream

agent = Agent(
    edge=getstream.Edge(),
    stt=bytedance.STT(),
    llm=openai.LLM(model="gpt-4o-mini"),
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
