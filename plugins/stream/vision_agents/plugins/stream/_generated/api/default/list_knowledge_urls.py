from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.knowledge_url import KnowledgeUrl
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
        "url": "/v1/agents/knowledge/urls",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[KnowledgeUrl] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = KnowledgeUrl.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[KnowledgeUrl]]:
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
) -> Response[Error | list[KnowledgeUrl]]:
    """The pages a knowledge base is kept filled from

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[KnowledgeUrl]]
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
) -> Error | list[KnowledgeUrl] | None:
    """The pages a knowledge base is kept filled from

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[KnowledgeUrl]
    """

    return sync_detailed(
        client=client,
        namespace=namespace,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    namespace: str | Unset = UNSET,
) -> Response[Error | list[KnowledgeUrl]]:
    """The pages a knowledge base is kept filled from

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[KnowledgeUrl]]
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
) -> Error | list[KnowledgeUrl] | None:
    """The pages a knowledge base is kept filled from

    Args:
        namespace (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[KnowledgeUrl]
    """

    return (
        await asyncio_detailed(
            client=client,
            namespace=namespace,
        )
    ).parsed
