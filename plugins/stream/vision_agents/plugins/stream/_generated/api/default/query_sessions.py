from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.session_page import SessionPage
from ...models.session_query import SessionQuery
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SessionQuery | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/query",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | SessionPage | None:
    if response.status_code == 200:
        response_200 = SessionPage.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | SessionPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SessionQuery | Unset = UNSET,
) -> Response[ErrorResponse | SessionPage]:
    """List or search the caller's sessions

     Three queries are supported, each over the sessions still running and the ones that ended:

    - every session, sorted by `updated_at`
    - a text search, `{"text": {"$q": "billing"}}`, sorted by `relevance`
    - one project's, `{"project_id": "health"}`, sorted by `updated_at`

    `agent`, `agent_id`, `config_id`, `user_id`, `modality`, `state`, `created_at` and `custom` narrow
    any of them. A backend gets its customer's sessions; an end user gets their own, whatever they ask
    for, and an anonymous caller who named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. There is no total:
    counting every conversation costs more than the page.

    Args:
        body (SessionQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SessionPage]
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
    body: SessionQuery | Unset = UNSET,
) -> ErrorResponse | SessionPage | None:
    """List or search the caller's sessions

     Three queries are supported, each over the sessions still running and the ones that ended:

    - every session, sorted by `updated_at`
    - a text search, `{"text": {"$q": "billing"}}`, sorted by `relevance`
    - one project's, `{"project_id": "health"}`, sorted by `updated_at`

    `agent`, `agent_id`, `config_id`, `user_id`, `modality`, `state`, `created_at` and `custom` narrow
    any of them. A backend gets its customer's sessions; an end user gets their own, whatever they ask
    for, and an anonymous caller who named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. There is no total:
    counting every conversation costs more than the page.

    Args:
        body (SessionQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SessionPage
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SessionQuery | Unset = UNSET,
) -> Response[ErrorResponse | SessionPage]:
    """List or search the caller's sessions

     Three queries are supported, each over the sessions still running and the ones that ended:

    - every session, sorted by `updated_at`
    - a text search, `{"text": {"$q": "billing"}}`, sorted by `relevance`
    - one project's, `{"project_id": "health"}`, sorted by `updated_at`

    `agent`, `agent_id`, `config_id`, `user_id`, `modality`, `state`, `created_at` and `custom` narrow
    any of them. A backend gets its customer's sessions; an end user gets their own, whatever they ask
    for, and an anonymous caller who named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. There is no total:
    counting every conversation costs more than the page.

    Args:
        body (SessionQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SessionPage]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SessionQuery | Unset = UNSET,
) -> ErrorResponse | SessionPage | None:
    """List or search the caller's sessions

     Three queries are supported, each over the sessions still running and the ones that ended:

    - every session, sorted by `updated_at`
    - a text search, `{"text": {"$q": "billing"}}`, sorted by `relevance`
    - one project's, `{"project_id": "health"}`, sorted by `updated_at`

    `agent`, `agent_id`, `config_id`, `user_id`, `modality`, `state`, `created_at` and `custom` narrow
    any of them. A backend gets its customer's sessions; an end user gets their own, whatever they ask
    for, and an anonymous caller who named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. There is no total:
    counting every conversation costs more than the page.

    Args:
        body (SessionQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SessionPage
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
