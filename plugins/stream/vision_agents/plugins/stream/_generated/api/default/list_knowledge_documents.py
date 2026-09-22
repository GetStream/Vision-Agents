from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.indexed_knowledge_document import IndexedKnowledgeDocument
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    namespace: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["namespace"] = namespace

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/knowledge/documents",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[IndexedKnowledgeDocument] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = IndexedKnowledgeDocument.from_dict(
                response_200_item_data
            )

            response_200.append(response_200_item)

        return response_200

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
) -> Response[Error | list[IndexedKnowledgeDocument]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    namespace: str | Unset = UNSET,
) -> Response[Error | list[IndexedKnowledgeDocument]]:
    """The documents a knowledge base was filled with

     What was posted to /v1/agents/knowledge or synced from an agent directory's knowledge folder, one
    entry per source. Pages are listed at /v1/agents/knowledge/urls.

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[IndexedKnowledgeDocument]]
    """

    kwargs = _get_kwargs(
        namespace=namespace,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    namespace: str | Unset = UNSET,
) -> Error | list[IndexedKnowledgeDocument] | None:
    """The documents a knowledge base was filled with

     What was posted to /v1/agents/knowledge or synced from an agent directory's knowledge folder, one
    entry per source. Pages are listed at /v1/agents/knowledge/urls.

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[IndexedKnowledgeDocument]
    """

    return sync_detailed(
        client=client,
        namespace=namespace,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    namespace: str | Unset = UNSET,
) -> Response[Error | list[IndexedKnowledgeDocument]]:
    """The documents a knowledge base was filled with

     What was posted to /v1/agents/knowledge or synced from an agent directory's knowledge folder, one
    entry per source. Pages are listed at /v1/agents/knowledge/urls.

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[IndexedKnowledgeDocument]]
    """

    kwargs = _get_kwargs(
        namespace=namespace,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    namespace: str | Unset = UNSET,
) -> Error | list[IndexedKnowledgeDocument] | None:
    """The documents a knowledge base was filled with

     What was posted to /v1/agents/knowledge or synced from an agent directory's knowledge folder, one
    entry per source. Pages are listed at /v1/agents/knowledge/urls.

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[IndexedKnowledgeDocument]
    """

    return (
        await asyncio_detailed(
            client=client,
            namespace=namespace,
        )
    ).parsed
