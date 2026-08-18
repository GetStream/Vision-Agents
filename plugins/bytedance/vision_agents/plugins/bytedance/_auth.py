"""Shared authentication helpers for ByteDance / BytePlus Seed Speech APIs.

All Seed Speech WebSocket services authenticate through HTTP headers on the
upgrade request. The new Volcengine / BytePlus console uses a single
``X-Api-Key``; the legacy console uses ``X-Api-App-Key`` + ``X-Api-Access-Key``.
"""

import os
import uuid
from dataclasses import dataclass
from typing import Optional

DEFAULT_WS_HOST = "wss://openspeech.bytedance.com"


@dataclass
class Credentials:
    """Resolved Seed Speech credentials.

    Either ``api_key`` (new console) or the ``app_key`` + ``access_key`` pair
    (legacy console) must be present.
    """

    api_key: Optional[str] = None
    app_key: Optional[str] = None
    access_key: Optional[str] = None

    @classmethod
    def resolve(
        cls,
        api_key: Optional[str] = None,
        app_key: Optional[str] = None,
        access_key: Optional[str] = None,
    ) -> "Credentials":
        """Resolve credentials from arguments, falling back to the environment.

        Env vars: ``BYTEDANCE_API_KEY`` (or ``BYTEPLUS_API_KEY``),
        ``BYTEDANCE_APP_KEY``, ``BYTEDANCE_ACCESS_KEY``.
        """
        api_key = (
            api_key
            or os.environ.get("BYTEDANCE_API_KEY")
            or os.environ.get("BYTEPLUS_API_KEY")
        )
        app_key = app_key or os.environ.get("BYTEDANCE_APP_KEY")
        access_key = access_key or os.environ.get("BYTEDANCE_ACCESS_KEY")

        if not api_key and not (app_key and access_key):
            raise ValueError(
                "ByteDance credentials missing: set BYTEDANCE_API_KEY (new console) "
                "or BYTEDANCE_APP_KEY + BYTEDANCE_ACCESS_KEY (legacy console)"
            )
        return cls(api_key=api_key, app_key=app_key, access_key=access_key)

    def headers(
        self, resource_id: str, *, connect_id: Optional[str] = None
    ) -> dict[str, str]:
        """Build the WebSocket handshake headers for a Seed Speech service.

        Args:
            resource_id: The ``X-Api-Resource-Id`` selecting the service/SKU.
            connect_id: Optional connection tracing ID; a UUID is generated when omitted.
        """
        headers = {
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": connect_id or str(uuid.uuid4()),
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        else:
            headers["X-Api-App-Key"] = self.app_key or ""
            headers["X-Api-Access-Key"] = self.access_key or ""
        return headers
