import os
from dataclasses import dataclass
from typing import Optional

from ._generated import AuthenticatedClient

URL_ENV = "STREAM_ACCELERATION_URL"
CUSTOMER_ENV = "STREAM_ACCELERATION_CUSTOMER_ID"
CUSTOMER_HEADER = "X-Customer-Id"
DEFAULT_URL = "http://localhost:8080"


@dataclass
class Backend:
    """Where the acceleration router is, and who is calling it.

    Attributes:
        url: The router's base URL. Defaults to ``STREAM_ACCELERATION_URL``.
        customer_id: The identity every request and every cost row is keyed by.
            Defaults to ``STREAM_ACCELERATION_CUSTOMER_ID``.
    """

    url: Optional[str] = None
    customer_id: Optional[str] = None

    def __post_init__(self):
        self.url = (self.url or os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.customer_id = self.customer_id or os.environ.get(CUSTOMER_ENV, "")
        if not self.customer_id:
            raise ValueError(
                f"a customer id is required; pass customer_id or set {CUSTOMER_ENV}"
            )

    @property
    def headers(self) -> dict[str, str]:
        """The headers every request to the router carries."""
        return {CUSTOMER_HEADER: str(self.customer_id)}

    def client(self) -> AuthenticatedClient:
        """An HTTP client for the generated API."""
        return AuthenticatedClient(
            base_url=str(self.url),
            token=str(self.customer_id),
            auth_header_name=CUSTOMER_HEADER,
            prefix="",
        )

    def socket(self, path: str) -> str:
        """The WebSocket URL for a path on the router."""
        base = str(self.url)
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :] + path
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :] + path
        return base + path
