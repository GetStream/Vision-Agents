import logging
from os import getenv
from typing import Any

import httpx

from .exceptions import LemonSliceSessionError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://lemonslice.com/api/liveai"


class LemonSliceClient:
    """REST API client for LemonSlice session management.

    Handles authentication and session creation with LemonSlice's
    self-managed API for real-time avatar generation.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_image_url: str | None = None,
        agent_prompt: str | None = None,
        idle_timeout: int | None = None,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        lemonslice_properties: dict[str, Any] | None = None,
    ):
        """Initialize the LemonSlice client.

        Args:
            agent_id: LemonSlice agent ID.
            agent_image_url: Custom agent image URL (368x560px recommended).
            agent_prompt: Prompt influencing avatar expressions and movements.
            idle_timeout: Session timeout in seconds.
            api_key: LemonSlice API key. Uses LEMONSLICE_API_KEY env var if not provided.
            base_url: LemonSlice API base URL.
            lemonslice_properties: Extra fields added to the session creation payload.
        """
        if not agent_id and not agent_image_url:
            raise ValueError("Either agent_id or agent_image_url must be provided.")

        self._api_key: str = api_key or getenv("LEMONSLICE_API_KEY") or ""
        if not self._api_key:
            raise ValueError(
                "LemonSlice API key required. Set LEMONSLICE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self._agent_id = agent_id
        self._agent_image_url = agent_image_url
        self._agent_prompt = agent_prompt
        self._idle_timeout = idle_timeout
        self._lemonslice_properties = lemonslice_properties or {}
        self._session_id: str | None = None
        self._http_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "X-API-Key": self._api_key,
                "Content-Type": "application/json",
            },
        )

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def create_session(
        self, call_id: str, call_type: str, token: str, api_key: str
    ) -> str:
        """Create a new LemonSlice avatar session.

        Args:
            call_id: Stream call ID the avatar should join.
            call_type: Stream call type (e.g. "default").
            token: Stream access token for the avatar participant.
            api_key: Stream API key.

        Returns:
            The created session ID.
        """
        payload: dict[str, object] = {
            **self._lemonslice_properties,
            "transport_type": "stream",
            "properties": {
                "call_id": call_id,
                "call_type": call_type,
                "token": token,
                "api_key": api_key,
            },
        }

        if self._agent_id:
            payload["agent_id"] = self._agent_id
        if self._agent_image_url:
            payload["agent_image_url"] = self._agent_image_url
        if self._agent_prompt:
            payload["agent_prompt"] = self._agent_prompt
        if self._idle_timeout is not None:
            payload["idle_timeout"] = self._idle_timeout

        response = await self._http_client.post("/sessions", json=payload)

        if response.status_code >= 400:
            raise LemonSliceSessionError(
                f"Failed to create session: {response.status_code} - {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        self._session_id = data.get("session_id")

        if not self._session_id:
            raise LemonSliceSessionError(
                f"Session creation returned no session_id: {data}"
            )

        logger.info(f"LemonSlice session created: {self._session_id}")
        return self._session_id

    async def close(self) -> None:
        """Clean up client resources."""
        try:
            await self._http_client.aclose()
        finally:
            self._session_id = None
            logger.debug("LemonSlice client closed")
