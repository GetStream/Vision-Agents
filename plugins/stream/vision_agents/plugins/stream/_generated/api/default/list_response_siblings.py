from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.list_response_siblings_of import ListResponseSiblingsOf
from ...models.response_sibling_page import ResponseSiblingPage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    response_id: str,
    *,
    of: ListResponseSiblingsOf | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_of: str | Unset = UNSET
    if not isinstance(of, Unset):
        json_of = of.value

    params["of"] = json_of

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/responses/{response_id}/siblings".format(
            id=quote(str(id), safe=""),
            response_id=quote(str(response_id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ResponseSiblingPage | None:
    if response.status_code == 200:
        response_200 = ResponseSiblingPage.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | ResponseSiblingPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
    of: ListResponseSiblingsOf | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | ResponseSiblingPage]:
    """The alternatives to one response: its regenerations, or the edits of its question

     What a ‹ i/n › control pages through. of=reply lists every answer to the response's question, the
    response's own included; of=question lists the questions asked at the same point of the
    conversation, the original and its edits, each as the first response that answered it. Oldest first,
    which is the order of their index.
    active marks the one the conversation goes through now. Reading a sibling's messages is
    getResponseBranch, and making it the conversation is activateResponse. A response whose messages
    carry no branch mark, such as a turn spoken on a call, has no siblings but itself.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): The response whose alternatives to list, itself among them.
        of (ListResponseSiblingsOf | Unset): reply lists the answers to the response's question;
            question lists the questions asked in its place. Omitted is reply.
        limit (int | Unset): Up to 100. Omitted is 25.
        cursor (str | Unset): The `next_cursor` of the previous page, sent with the same filters.
            Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ResponseSiblingPage]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
        of=of,
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
    of: ListResponseSiblingsOf | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | ResponseSiblingPage | None:
    """The alternatives to one response: its regenerations, or the edits of its question

     What a ‹ i/n › control pages through. of=reply lists every answer to the response's question, the
    response's own included; of=question lists the questions asked at the same point of the
    conversation, the original and its edits, each as the first response that answered it. Oldest first,
    which is the order of their index.
    active marks the one the conversation goes through now. Reading a sibling's messages is
    getResponseBranch, and making it the conversation is activateResponse. A response whose messages
    carry no branch mark, such as a turn spoken on a call, has no siblings but itself.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): The response whose alternatives to list, itself among them.
        of (ListResponseSiblingsOf | Unset): reply lists the answers to the response's question;
            question lists the questions asked in its place. Omitted is reply.
        limit (int | Unset): Up to 100. Omitted is 25.
        cursor (str | Unset): The `next_cursor` of the previous page, sent with the same filters.
            Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ResponseSiblingPage
    """

    return sync_detailed(
        id=id,
        response_id=response_id,
        client=client,
        of=of,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
    of: ListResponseSiblingsOf | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | ResponseSiblingPage]:
    """The alternatives to one response: its regenerations, or the edits of its question

     What a ‹ i/n › control pages through. of=reply lists every answer to the response's question, the
    response's own included; of=question lists the questions asked at the same point of the
    conversation, the original and its edits, each as the first response that answered it. Oldest first,
    which is the order of their index.
    active marks the one the conversation goes through now. Reading a sibling's messages is
    getResponseBranch, and making it the conversation is activateResponse. A response whose messages
    carry no branch mark, such as a turn spoken on a call, has no siblings but itself.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): The response whose alternatives to list, itself among them.
        of (ListResponseSiblingsOf | Unset): reply lists the answers to the response's question;
            question lists the questions asked in its place. Omitted is reply.
        limit (int | Unset): Up to 100. Omitted is 25.
        cursor (str | Unset): The `next_cursor` of the previous page, sent with the same filters.
            Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ResponseSiblingPage]
    """

    kwargs = _get_kwargs(
        id=id,
        response_id=response_id,
        of=of,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    response_id: str,
    *,
    client: AuthenticatedClient | Client,
    of: ListResponseSiblingsOf | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | ResponseSiblingPage | None:
    """The alternatives to one response: its regenerations, or the edits of its question

     What a ‹ i/n › control pages through. of=reply lists every answer to the response's question, the
    response's own included; of=question lists the questions asked at the same point of the
    conversation, the original and its edits, each as the first response that answered it. Oldest first,
    which is the order of their index.
    active marks the one the conversation goes through now. Reading a sibling's messages is
    getResponseBranch, and making it the conversation is activateResponse. A response whose messages
    carry no branch mark, such as a turn spoken on a call, has no siblings but itself.

    Args:
        id (str): The session, as returned when it was created.
        response_id (str): The response whose alternatives to list, itself among them.
        of (ListResponseSiblingsOf | Unset): reply lists the answers to the response's question;
            question lists the questions asked in its place. Omitted is reply.
        limit (int | Unset): Up to 100. Omitted is 25.
        cursor (str | Unset): The `next_cursor` of the previous page, sent with the same filters.
            Omitted is the first page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ResponseSiblingPage
    """

    return (
        await asyncio_detailed(
            id=id,
            response_id=response_id,
            client=client,
            of=of,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
