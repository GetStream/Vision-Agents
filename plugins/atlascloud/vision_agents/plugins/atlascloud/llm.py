import os

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

ATLAS_CLOUD_BASE_URL = "https://api.atlascloud.ai/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4-pro"


class AtlasCloudLLM(ChatCompletionsLLM):
    """Atlas Cloud LLM using the OpenAI-compatible Chat Completions API."""

    provider_name = "atlascloud"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = ATLAS_CLOUD_BASE_URL,
        client: AsyncOpenAI | None = None,
        tools_max_rounds: int = 3,
    ) -> None:
        """Initialize the Atlas Cloud LLM."""
        resolved_key = (
            api_key
            or os.environ.get("ATLASCLOUD_API_KEY")
            or os.environ.get("ATLAS_CLOUD_API_KEY")
        )
        if client is None and not resolved_key:
            raise ValueError(
                "ATLASCLOUD_API_KEY or ATLAS_CLOUD_API_KEY env var, "
                "or api_key parameter required for Atlas Cloud LLM"
            )

        super().__init__(
            model=model,
            api_key=resolved_key,
            base_url=base_url,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
