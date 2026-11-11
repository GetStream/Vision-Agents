import pytest
from openai import AsyncOpenAI
from vision_agents.plugins.atlascloud import LLM


class TestAtlasCloudLLM:
    """Unit tests for Atlas Cloud LLM configuration."""

    def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
        monkeypatch.delenv("ATLAS_CLOUD_API_KEY", raising=False)

        with pytest.raises(ValueError, match="ATLASCLOUD_API_KEY"):
            LLM()

    @pytest.mark.parametrize("env_name", ["ATLASCLOUD_API_KEY", "ATLAS_CLOUD_API_KEY"])
    async def test_api_key_env_aliases(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str
    ) -> None:
        monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
        monkeypatch.delenv("ATLAS_CLOUD_API_KEY", raising=False)
        monkeypatch.setenv(env_name, "test-key")

        llm = LLM()

        assert llm.model == "deepseek-ai/deepseek-v4-pro"
        assert str(llm._client.base_url) == "https://api.atlascloud.ai/v1/"
        await llm.close()

    async def test_explicit_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ATLASCLOUD_API_KEY", "env-key")

        llm = LLM(
            api_key="explicit-key",
            model="qwen/qwen3.5-27b",
            base_url="https://example.com/v1",
        )

        assert llm.model == "qwen/qwen3.5-27b"
        assert llm._client.api_key == "explicit-key"
        assert str(llm._client.base_url) == "https://example.com/v1/"
        await llm.close()

    async def test_preconfigured_client_does_not_require_env_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ATLASCLOUD_API_KEY", raising=False)
        monkeypatch.delenv("ATLAS_CLOUD_API_KEY", raising=False)
        client = AsyncOpenAI(
            api_key="client-key",
            base_url="https://example.com/v1",
        )

        llm = LLM(client=client)

        assert llm._client is client
        await llm.close()
