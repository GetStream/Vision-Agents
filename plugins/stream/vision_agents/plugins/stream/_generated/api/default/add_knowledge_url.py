from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.knowledge_url import KnowledgeUrl
from ...models.knowledge_url_request import KnowledgeUrlRequest
from ...types import Response


def _get_kwargs(
    *,
    body: KnowledgeUrlRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/knowledge/urls",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | KnowledgeUrl | None:
    if response.status_code == 201:
        response_201 = KnowledgeUrl.from_dict(response.json())

        return response_201

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | KnowledgeUrl]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: KnowledgeUrlRequest,
) -> Response[Error | KnowledgeUrl]:
    """Keep a knowledge base filled from a page

     Posting a document is a thing that happens once; a url is a subscription, because the page behind it
    changes and nobody re-posts it. The page is fetched here, turned into markdown, cut into passages
    the same way a document is, and written under the url so a later read replaces it rather than adding
    a second copy.
    The fetch happens before this answers and a live crawl takes seconds, so this is slower than the
    endpoints around it. A page that could not be read is still stored, in the failed state with the
    reason on it, rather than refused and forgotten.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (KnowledgeUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | KnowledgeUrl]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: KnowledgeUrlRequest,
) -> Error | KnowledgeUrl | None:
    """Keep a knowledge base filled from a page

     Posting a document is a thing that happens once; a url is a subscription, because the page behind it
    changes and nobody re-posts it. The page is fetched here, turned into markdown, cut into passages
    the same way a document is, and written under the url so a later read replaces it rather than adding
    a second copy.
    The fetch happens before this answers and a live crawl takes seconds, so this is slower than the
    endpoints around it. A page that could not be read is still stored, in the failed state with the
    reason on it, rather than refused and forgotten.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (KnowledgeUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | KnowledgeUrl
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: KnowledgeUrlRequest,
) -> Response[Error | KnowledgeUrl]:
    """Keep a knowledge base filled from a page

     Posting a document is a thing that happens once; a url is a subscription, because the page behind it
    changes and nobody re-posts it. The page is fetched here, turned into markdown, cut into passages
    the same way a document is, and written under the url so a later read replaces it rather than adding
    a second copy.
    The fetch happens before this answers and a live crawl takes seconds, so this is slower than the
    endpoints around it. A page that could not be read is still stored, in the failed state with the
    reason on it, rather than refused and forgotten.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (KnowledgeUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | KnowledgeUrl]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: KnowledgeUrlRequest,
) -> Error | KnowledgeUrl | None:
    """Keep a knowledge base filled from a page

     Posting a document is a thing that happens once; a url is a subscription, because the page behind it
    changes and nobody re-posts it. The page is fetched here, turned into markdown, cut into passages
    the same way a document is, and written under the url so a later read replaces it rather than adding
    a second copy.
    The fetch happens before this answers and a live crawl takes seconds, so this is slower than the
    endpoints around it. A page that could not be read is still stored, in the failed state with the
    reason on it, rather than refused and forgotten.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (KnowledgeUrlRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | KnowledgeUrl
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
