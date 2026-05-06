"""Inworld AI LLM router via the OpenAI-compatible chat completions endpoint.

Inworld's `/v1/chat/completions` is wire-compatible with OpenAI's Chat
Completions API and routes upstream across providers (OpenAI, Anthropic,
Google, etc.) with auto-selection, fallbacks, and traffic splitting.
This class is a thin wrapper over the openai plugin's ``ChatCompletionsLLM``
that defaults the base URL and API-key env var, and exposes Inworld
router-specific routing fields via constructor kwargs.
"""

import os
from typing import Any, Optional

from openai import AsyncOpenAI
from vision_agents.plugins.openai import ChatCompletionsLLM

INWORLD_BASE_URL = "https://api.inworld.ai/v1"
PLUGIN_NAME = "inworld"


class LLM(ChatCompletionsLLM):
    """Inworld LLM router (text chat completions).

    The ``model`` argument accepts:

    - ``"inworld/<router-id>"`` for routers configured in the Inworld portal
    - ``"<provider>/<model-id>"`` (e.g. ``"openai/gpt-4o-mini"``)
    - ``"auto"`` for server-side selection across providers

    Examples:

        from vision_agents.plugins import inworld

        llm = inworld.LLM(model="auto", sort_by=["latency"])

        # Single-model with fallbacks
        llm = inworld.LLM(
            model="openai/gpt-4o-mini",
            fallback_models=["anthropic/claude-sonnet-4-5"],
            ttft_timeout="500ms",
        )
    """

    provider_name = PLUGIN_NAME

    def __init__(
        self,
        model: str = "auto",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        tools_max_rounds: int = 3,
        *,
        fallback_models: Optional[list[str]] = None,
        ignore_models: Optional[list[str]] = None,
        sort_by: Optional[list[str]] = None,
        ttft_timeout: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        web_search: bool = False,
        web_search_options: Optional[dict[str, Any]] = None,
        compression_aggressiveness: Optional[float] = None,
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the Inworld LLM router client.

        Args:
            model: Inworld model id. Use ``"auto"``, ``"<provider>/<model>"``,
                or ``"inworld/<router-id>"``.
            api_key: Inworld API key. Falls back to ``INWORLD_API_KEY`` env var.
            base_url: Override the API base URL. Defaults to
                ``https://api.inworld.ai/v1``.
            client: Optional pre-built ``AsyncOpenAI`` client. If provided,
                ``api_key`` and ``base_url`` are ignored.
            tools_max_rounds: Max calling rounds for multi-hop tool calls.
            fallback_models: Ordered list of provider/model ids to try if
                the primary fails.
            ignore_models: Provider/model ids to exclude from auto selection.
            sort_by: Sort metrics for ``model="auto"``, sent as an array of
                strings to ``extra_body.sort``. Accepted values per Inworld
                docs: ``"price"``, ``"latency"``, ``"throughput"``,
                ``"intelligence"``, ``"math"``, ``"coding"``. Multiple
                metrics rank with tiebreakers.
            ttft_timeout: Trigger fallback if first token does not arrive in
                this duration (e.g. ``"500ms"``, ``"1s"``). Inworld minimum
                is ``"300ms"``.
            metadata: Free-form metadata dict, used by router CEL expressions
                for conditional routing (e.g. ``{"tier": "premium"}``).
            web_search: Enable upstream web search grounding.
            web_search_options: Provider-specific web search options.
            compression_aggressiveness: 0–1. Applied to the system message
                via Inworld's ``compression`` field. Reduces input tokens
                without changing semantics, lowering TTFT for long prompts.
            extra_body: Raw escape hatch merged into the request's
                ``extra_body`` after the helpers above (overrides them).
        """
        super().__init__(
            model=model,
            api_key=api_key or os.environ.get("INWORLD_API_KEY"),
            base_url=base_url or INWORLD_BASE_URL,
            client=client,
            tools_max_rounds=tools_max_rounds,
        )
        self._compression_aggressiveness = compression_aggressiveness
        self._extra_body = self._build_extra_body(
            fallback_models=fallback_models,
            ignore_models=ignore_models,
            sort_by=sort_by,
            ttft_timeout=ttft_timeout,
            metadata=metadata,
            web_search=web_search,
            web_search_options=web_search_options,
            override=extra_body,
        )

    @staticmethod
    def _build_extra_body(
        *,
        fallback_models: Optional[list[str]],
        ignore_models: Optional[list[str]],
        sort_by: Optional[list[str]],
        ttft_timeout: Optional[str],
        metadata: Optional[dict[str, Any]],
        web_search: bool,
        web_search_options: Optional[dict[str, Any]],
        override: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if fallback_models:
            body["models"] = list(fallback_models)
        if ignore_models:
            body["ignore"] = list(ignore_models)
        if sort_by:
            body["sort"] = list(sort_by)
        if ttft_timeout:
            body.setdefault("fallback", {})["ttft_timeout"] = ttft_timeout
        if metadata:
            body["metadata"] = dict(metadata)
        if web_search:
            body["web_search"] = True
            if web_search_options:
                body["web_search_options"] = dict(web_search_options)
        if override:
            body.update(override)
        return body

    def _extra_request_kwargs(self) -> dict[str, Any]:
        return {"extra_body": self._extra_body} if self._extra_body else {}

    async def _build_model_request(self) -> list[dict]:
        messages = await super()._build_model_request()
        if self._compression_aggressiveness is not None:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["compression"] = {
                        "aggressiveness": self._compression_aggressiveness
                    }
                    break
        return messages
