"""What the router answers instead of doing something, as one exception."""

import json
from typing import Any, Awaitable, Mapping, Optional

import httpx

# REQUEST_ID_HEADER names the request in the router's logs, which is what a person quotes
# to support: a 500 says nothing more than "something went wrong".
REQUEST_ID_HEADER = "X-Request-Id"


class RouterError(RuntimeError):
    """What the router said it would not do, raised where it was asked.

    A refusal made here, before anything was sent, carries the message alone.

    Attributes:
        status: The HTTP status the router answered with.
        type: The kind of failure, which decides the status: ``invalid_request``,
            ``not_found``, ``rate_limited``, ``internal`` and the rest. None when the body
            was not the router's own, such as a proxy's error page.
        code: What to branch on, such as ``not_configured`` or ``validation_failed``. New
            codes appear, so an unknown one is not an error.
        doc_url: Where the code is explained.
        request_id: The ``X-Request-Id`` the router answered with.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        type: Optional[str] = None,
        code: Optional[str] = None,
        doc_url: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.status = status
        self.type = type
        self.code = code
        self.doc_url = doc_url
        self.request_id = request_id


def refusal(status: int, headers: Mapping[str, str], body: bytes) -> RouterError:
    """Read one answer outside 2xx as the RouterError it stands for.

    The router answers ``{"error": {"message", "type", "code", "doc_url"}}``. Anything
    else, a proxy's HTML or an older router's ``{"error": "..."}`` included, is kept as
    the text it was, so the failure is never replaced by a failure to decode it.
    """
    request_id = headers.get(REQUEST_ID_HEADER)
    detail = _envelope(body)
    if detail is None:
        text = body.decode(errors="replace").strip()
        return RouterError(
            text or _status_text(status), status=status, request_id=request_id
        )
    return RouterError(
        detail["message"],
        status=status,
        type=_string(detail.get("type")),
        code=_string(detail.get("code")),
        doc_url=_string(detail.get("doc_url")),
        request_id=request_id,
    )


def raise_refusal(response: httpx.Response) -> Optional[Awaitable[None]]:
    """An httpx response hook raising RouterError for any answer outside 2xx.

    It runs before the generated client parses the body, which decodes every documented
    failure as the envelope and fails on anything else. httpx awaits what a hook returns
    on its async client and ignores it on the sync one, so this one serves both.
    """
    if isinstance(response.stream, httpx.AsyncByteStream):
        return _raise_async_refusal(response)
    if not response.is_success:
        response.read()
        raise refusal(response.status_code, response.headers, response.content)
    return None


async def _raise_async_refusal(response: httpx.Response) -> None:
    if not response.is_success:
        await response.aread()
        raise refusal(response.status_code, response.headers, response.content)


def _envelope(body: bytes) -> Optional[dict[str, Any]]:
    """The envelope's ``error`` object, or None for a body that is not the envelope."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    detail = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail
    return None


def _string(value: object) -> Optional[str]:
    return value if isinstance(value, str) else None


def _status_text(status: int) -> str:
    """What to say about an answer that came with no body."""
    phrase = httpx.codes.get_reason_phrase(status)
    return f"the router answered {status} {phrase}".rstrip()
