# Cheaper Inference Plugin for Vision Agents

The Cheaper Inference plugin adds [Cheaper Inference](https://cheaperinference.com) as an LLM provider for [Vision Agents](https://visionagents.ai). Cheaper Inference is an OpenAI-compatible LLM gateway. The plugin uses the Chat Completions API at `https://api.cheaperinference.com/v1`, so it inherits the streaming and tool-calling behavior of the OpenAI Chat Completions plugin.

## Installation

```bash
uv add "vision-agents-plugins-cheaperinference"
```

## Configuration

Set your Cheaper Inference API key (get one at <https://cheaperinference.com/signup>):

```bash
export CHEAPER_INFERENCE_API_KEY=...
```

The plugin also honors `CHEAPER_INFERENCE_BASE_URL` to override the endpoint.

## Models

Model ids are bare. The default is `gpt-5.4-mini`. Other examples: `gpt-5.4`, `claude-sonnet-5`, `gemini-3.1-pro`, `deepseek-v4-flash`.
The full list is at <https://cheaperinference.com/markets> and `GET https://api.cheaperinference.com/v1/models`.

## Usage

```python
from vision_agents.plugins import cheaperinference

llm = cheaperinference.LLM()  # defaults to gpt-5.4-mini
llm = cheaperinference.LLM(model="claude-sonnet-5")
```
