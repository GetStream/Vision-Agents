import logging
from dataclasses import dataclass
from typing import Self

import httpx

from .exceptions import LiveAvatarAPIError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.liveavatar.com"


@dataclass
class SessionToken:
    session_id: str
    session_token: str


@dataclass
class Session:
    session_id: str
    livekit_url: str
    livekit_agent_token: str
    livekit_client_token: str
    ws_url: str
    max_session_duration: int | None = None


class LiveAvatarClient:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("LiveAvatar API key required")
        self._api_key = api_key
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={"X-API-KEY": api_key},
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def create_session_token(
        self,
        avatar_id: str,
        *,
        is_sandbox: bool = True,
        max_session_duration: int | None = None,
        video_quality: str = "high",
        video_encoding: str = "H264",
    ) -> SessionToken:
        body: dict[str, object] = {
            "mode": "LITE",
            "avatar_id": avatar_id,
            "is_sandbox": is_sandbox,
            "video_settings": {
                "quality": video_quality,
                "encoding": video_encoding,
            },
        }
        if max_session_duration is not None:
            body["max_session_duration"] = max_session_duration

        resp = await self._http.post("/v1/sessions/token", json=body)
        self._raise_for_status(resp)
        data = resp.json()["data"]
        return SessionToken(
            session_id=data["session_id"],
            session_token=data["session_token"],
        )

    async def start_session(self, session_token: str) -> Session:
        resp = await self._http.post(
            "/v1/sessions/start",
            headers={"Authorization": f"Bearer {session_token}"},
            json={},
        )
        self._raise_for_status(resp)
        data = resp.json()["data"]
        return Session(
            session_id=data["session_id"],
            livekit_url=data["livekit_url"],
            livekit_agent_token=data["livekit_agent_token"],
            livekit_client_token=data["livekit_client_token"],
            ws_url=data["ws_url"],
            max_session_duration=data.get("max_session_duration"),
        )

    async def stop_session(
        self,
        *,
        session_id: str,
        reason: str = "USER_CLOSED",
    ) -> None:
        resp = await self._http.post(
            "/v1/sessions/stop",
            json={"session_id": session_id, "reason": reason},
        )
        self._raise_for_status(resp)

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise LiveAvatarAPIError(
                f"{e.request.method} {e.request.url} -> {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
                body=resp.text,
            ) from e
