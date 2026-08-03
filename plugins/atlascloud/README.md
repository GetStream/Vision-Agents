# Atlas Cloud LLM Plugin

OpenAI-compatible Atlas Cloud LLM integration for Vision Agents.

## Installation

```bash
uv add "vision-agents[atlascloud]"
```

## Usage

Set `ATLASCLOUD_API_KEY` or `ATLAS_CLOUD_API_KEY`, then create the LLM:

```python
from vision_agents.plugins import atlascloud

llm = atlascloud.LLM(model="deepseek-ai/deepseek-v4-pro")
```

The default endpoint is `https://api.atlascloud.ai/v1`. Pass `base_url` to
target another OpenAI-compatible endpoint.
