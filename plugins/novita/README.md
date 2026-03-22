# Novita AI Plugin for Vision Agents

Provides LLM integration with [Novita AI](https://novita.ai) using its OpenAI-compatible API endpoint.

## Installation

```bash
pip install vision-agents-plugins-novita
```

## Usage

```python
from vision_agents.plugins import novita

llm = novita.LLM(model="moonshotai/kimi-k2.5")
```

## Configuration

Set the `NOVITA_API_KEY` environment variable to your Novita AI API key.

## Available Models

| Model ID | Context | Architecture |
|----------|---------|--------------|
| `moonshotai/kimi-k2.5` (default) | 262,144 | MoE |
| `zai-org/glm-5` | 202,800 | MoE |
| `minimax/minimax-m2.5` | 204,800 | MoE |
