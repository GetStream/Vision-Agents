from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.ingest_knowledge_request import IngestKnowledgeRequest
from ...models.ingested_knowledge import IngestedKnowledge
from ...types import Response


def _get_kwargs(
    *,
    body: IngestKnowledgeRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/knowledge",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | IngestedKnowledge | None:
    if response.status_code == 200:
        response_200 = IngestedKnowledge.from_dict(response.json())

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
) -> Response[Error | IngestedKnowledge]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: IngestKnowledgeRequest,
) -> Response[Error | IngestedKnowledge]:
    """Fill a knowledge base with what the business wrote down

     The writing half of the lookup a config's knowledge_namespace gives an agent. Each document is cut
    into passages here rather than by the caller, so a file read off disk by the command and one posted
    by an SDK are cut the same way and can replace each other.
    Passages are keyed by the source and the position they came from, so posting a document again after
    editing it replaces that document's passages rather than leaving two versions of them to be found. A
    namespace is never shared between agents, and the caller names it.

    Args:
        body (IngestKnowledgeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | IngestedKnowledge]
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
    body: IngestKnowledgeRequest,
) -> Error | IngestedKnowledge | None:
    """Fill a knowledge base with what the business wrote down

     The writing half of the lookup a config's knowledge_namespace gives an agent. Each document is cut
    into passages here rather than by the caller, so a file read off disk by the command and one posted
    by an SDK are cut the same way and can replace each other.
    Passages are keyed by the source and the position they came from, so posting a document again after
    editing it replaces that document's passages rather than leaving two versions of them to be found. A
    namespace is never shared between agents, and the caller names it.

    Args:
        body (IngestKnowledgeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | IngestedKnowledge
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: IngestKnowledgeRequest,
) -> Response[Error | IngestedKnowledge]:
    """Fill a knowledge base with what the business wrote down

     The writing half of the lookup a config's knowledge_namespace gives an agent. Each document is cut
    into passages here rather than by the caller, so a file read off disk by the command and one posted
    by an SDK are cut the same way and can replace each other.
    Passages are keyed by the source and the position they came from, so posting a document again after
    editing it replaces that document's passages rather than leaving two versions of them to be found. A
    namespace is never shared between agents, and the caller names it.

    Args:
        body (IngestKnowledgeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | IngestedKnowledge]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: IngestKnowledgeRequest,
) -> Error | IngestedKnowledge | None:
    """Fill a knowledge base with what the business wrote down

     The writing half of the lookup a config's knowledge_namespace gives an agent. Each document is cut
    into passages here rather than by the caller, so a file read off disk by the command and one posted
    by an SDK are cut the same way and can replace each other.
    Passages are keyed by the source and the position they came from, so posting a document again after
    editing it replaces that document's passages rather than leaving two versions of them to be found. A
    namespace is never shared between agents, and the caller names it.

    Args:
        body (IngestKnowledgeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | IngestedKnowledge
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
