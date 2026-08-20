"""HTTP client for the voicebench world server."""

import json
import os
import urllib.error
import urllib.request
from typing import Any


class WorldClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.environ.get("VOICEBENCH_WORLD_URL") or "http://127.0.0.1:8090"
        ).rstrip("/")

    def call(self, tool: str, **args: Any) -> dict[str, Any]:
        payload = json.dumps(args).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/session/tools/{tool}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {"error": f"HTTP {exc.code}: {detail}"}
        if not body:
            return {}
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
