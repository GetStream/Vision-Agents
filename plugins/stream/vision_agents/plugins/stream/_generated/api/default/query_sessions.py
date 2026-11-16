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

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

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

     Lists the caller's sessions, the ones still running and the ones that ended, narrowed by a filter
    and in the order of a sort.

    A filter is an object of fields. The keys of one object are ANDed, and `$and` and `$or` each take a
    list of filters, so conditions combine and nest:

    - `project_id`: a value, `{"$eq": value}` or `{"$in": [values]}`
    - `agent`: a value or `{"$eq": value}`, the agent name the session was opened against
    - `text`: `{"$q": "words"}`, full text over the title, description, project and agent name
    - `review_score`: a number from 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt` and `$lte`; a server-side
    caller's only
    - `user_id`: a value or `{"$eq": value}`; a server-side caller's only
    - `config_id`: a value or `{"$eq": value}`, the agent config the session ran under, which a rename
    leaves alone
    - `state`: `live` or `ended`, as a value or `{"$eq": value}`
    - `created_at`: `{"$gte": time, "$lt": time}`, either bound alone or both, as RFC 3339 times
    - `custom`: an object of string labels the session's `custom` holds, every one of them

    Nothing else can be filtered on: any other key or operator is a 400 that names it. `$and` and `$or`
    take 1 to 10 filters and nest 3 deep, `$in` takes up to 50 values, and a filter holds at most 30
    field conditions. `text` appears once, at the top level or in a top-level `$and`, and never inside
    `$or`; `user_id` is placed the same way.

    Every session, pinned first:

    ```json
    {"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}
    ```

    One agent's sessions in either of two projects:

    ```json
    {"filter": {"agent": "support", "$or": [{"project_id": "health"}, {"project_id": "billing"}]}}
    ```

    A search within one project:

    ```json
    {"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}
    ```

    A backend's badly reviewed sessions and everything in two projects, the lowest score first:

    ```json
    {"filter": {"$or": [{"review_score": {"$lte": 2}}, {"project_id": {"$in": ["vip", "pilot"]}}]},
    "sort": [{"field": "review_score", "direction": 1}]}
    ```

    `sort` is `last_message_at` (the default, descending: when the agent last answered, or when the
    session began), `pinned` followed by `last_message_at`, `review_score` for a server-side caller,
    with the sessions nobody reviewed last, or `relevance`, which is how a `text` search is sorted and
    the only way it is. A session is pinned and unpinned by updating it.

    A backend gets its customer's sessions; an end user gets their own, whatever the filter says: an
    `$or` chooses among the sessions the caller may see and never adds to them. An anonymous caller who
    named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. Pages are followed
    by `next_cursor`, sent back with the same filter and sort. There is no total: counting every
    conversation costs more than the page.

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

     Lists the caller's sessions, the ones still running and the ones that ended, narrowed by a filter
    and in the order of a sort.

    A filter is an object of fields. The keys of one object are ANDed, and `$and` and `$or` each take a
    list of filters, so conditions combine and nest:

    - `project_id`: a value, `{"$eq": value}` or `{"$in": [values]}`
    - `agent`: a value or `{"$eq": value}`, the agent name the session was opened against
    - `text`: `{"$q": "words"}`, full text over the title, description, project and agent name
    - `review_score`: a number from 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt` and `$lte`; a server-side
    caller's only
    - `user_id`: a value or `{"$eq": value}`; a server-side caller's only
    - `config_id`: a value or `{"$eq": value}`, the agent config the session ran under, which a rename
    leaves alone
    - `state`: `live` or `ended`, as a value or `{"$eq": value}`
    - `created_at`: `{"$gte": time, "$lt": time}`, either bound alone or both, as RFC 3339 times
    - `custom`: an object of string labels the session's `custom` holds, every one of them

    Nothing else can be filtered on: any other key or operator is a 400 that names it. `$and` and `$or`
    take 1 to 10 filters and nest 3 deep, `$in` takes up to 50 values, and a filter holds at most 30
    field conditions. `text` appears once, at the top level or in a top-level `$and`, and never inside
    `$or`; `user_id` is placed the same way.

    Every session, pinned first:

    ```json
    {"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}
    ```

    One agent's sessions in either of two projects:

    ```json
    {"filter": {"agent": "support", "$or": [{"project_id": "health"}, {"project_id": "billing"}]}}
    ```

    A search within one project:

    ```json
    {"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}
    ```

    A backend's badly reviewed sessions and everything in two projects, the lowest score first:

    ```json
    {"filter": {"$or": [{"review_score": {"$lte": 2}}, {"project_id": {"$in": ["vip", "pilot"]}}]},
    "sort": [{"field": "review_score", "direction": 1}]}
    ```

    `sort` is `last_message_at` (the default, descending: when the agent last answered, or when the
    session began), `pinned` followed by `last_message_at`, `review_score` for a server-side caller,
    with the sessions nobody reviewed last, or `relevance`, which is how a `text` search is sorted and
    the only way it is. A session is pinned and unpinned by updating it.

    A backend gets its customer's sessions; an end user gets their own, whatever the filter says: an
    `$or` chooses among the sessions the caller may see and never adds to them. An anonymous caller who
    named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. Pages are followed
    by `next_cursor`, sent back with the same filter and sort. There is no total: counting every
    conversation costs more than the page.

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

     Lists the caller's sessions, the ones still running and the ones that ended, narrowed by a filter
    and in the order of a sort.

    A filter is an object of fields. The keys of one object are ANDed, and `$and` and `$or` each take a
    list of filters, so conditions combine and nest:

    - `project_id`: a value, `{"$eq": value}` or `{"$in": [values]}`
    - `agent`: a value or `{"$eq": value}`, the agent name the session was opened against
    - `text`: `{"$q": "words"}`, full text over the title, description, project and agent name
    - `review_score`: a number from 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt` and `$lte`; a server-side
    caller's only
    - `user_id`: a value or `{"$eq": value}`; a server-side caller's only
    - `config_id`: a value or `{"$eq": value}`, the agent config the session ran under, which a rename
    leaves alone
    - `state`: `live` or `ended`, as a value or `{"$eq": value}`
    - `created_at`: `{"$gte": time, "$lt": time}`, either bound alone or both, as RFC 3339 times
    - `custom`: an object of string labels the session's `custom` holds, every one of them

    Nothing else can be filtered on: any other key or operator is a 400 that names it. `$and` and `$or`
    take 1 to 10 filters and nest 3 deep, `$in` takes up to 50 values, and a filter holds at most 30
    field conditions. `text` appears once, at the top level or in a top-level `$and`, and never inside
    `$or`; `user_id` is placed the same way.

    Every session, pinned first:

    ```json
    {"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}
    ```

    One agent's sessions in either of two projects:

    ```json
    {"filter": {"agent": "support", "$or": [{"project_id": "health"}, {"project_id": "billing"}]}}
    ```

    A search within one project:

    ```json
    {"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}
    ```

    A backend's badly reviewed sessions and everything in two projects, the lowest score first:

    ```json
    {"filter": {"$or": [{"review_score": {"$lte": 2}}, {"project_id": {"$in": ["vip", "pilot"]}}]},
    "sort": [{"field": "review_score", "direction": 1}]}
    ```

    `sort` is `last_message_at` (the default, descending: when the agent last answered, or when the
    session began), `pinned` followed by `last_message_at`, `review_score` for a server-side caller,
    with the sessions nobody reviewed last, or `relevance`, which is how a `text` search is sorted and
    the only way it is. A session is pinned and unpinned by updating it.

    A backend gets its customer's sessions; an end user gets their own, whatever the filter says: an
    `$or` chooses among the sessions the caller may see and never adds to them. An anonymous caller who
    named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. Pages are followed
    by `next_cursor`, sent back with the same filter and sort. There is no total: counting every
    conversation costs more than the page.

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

     Lists the caller's sessions, the ones still running and the ones that ended, narrowed by a filter
    and in the order of a sort.

    A filter is an object of fields. The keys of one object are ANDed, and `$and` and `$or` each take a
    list of filters, so conditions combine and nest:

    - `project_id`: a value, `{"$eq": value}` or `{"$in": [values]}`
    - `agent`: a value or `{"$eq": value}`, the agent name the session was opened against
    - `text`: `{"$q": "words"}`, full text over the title, description, project and agent name
    - `review_score`: a number from 1 to 5, or `$eq`, `$gt`, `$gte`, `$lt` and `$lte`; a server-side
    caller's only
    - `user_id`: a value or `{"$eq": value}`; a server-side caller's only
    - `config_id`: a value or `{"$eq": value}`, the agent config the session ran under, which a rename
    leaves alone
    - `state`: `live` or `ended`, as a value or `{"$eq": value}`
    - `created_at`: `{"$gte": time, "$lt": time}`, either bound alone or both, as RFC 3339 times
    - `custom`: an object of string labels the session's `custom` holds, every one of them

    Nothing else can be filtered on: any other key or operator is a 400 that names it. `$and` and `$or`
    take 1 to 10 filters and nest 3 deep, `$in` takes up to 50 values, and a filter holds at most 30
    field conditions. `text` appears once, at the top level or in a top-level `$and`, and never inside
    `$or`; `user_id` is placed the same way.

    Every session, pinned first:

    ```json
    {"sort": [{"field": "pinned", "direction": -1}, {"field": "last_message_at", "direction": -1}]}
    ```

    One agent's sessions in either of two projects:

    ```json
    {"filter": {"agent": "support", "$or": [{"project_id": "health"}, {"project_id": "billing"}]}}
    ```

    A search within one project:

    ```json
    {"filter": {"text": {"$q": "refund"}, "project_id": "billing"}}
    ```

    A backend's badly reviewed sessions and everything in two projects, the lowest score first:

    ```json
    {"filter": {"$or": [{"review_score": {"$lte": 2}}, {"project_id": {"$in": ["vip", "pilot"]}}]},
    "sort": [{"field": "review_score", "direction": 1}]}
    ```

    `sort` is `last_message_at` (the default, descending: when the agent last answered, or when the
    session began), `pinned` followed by `last_message_at`, `review_score` for a server-side caller,
    with the sessions nobody reviewed last, or `relevance`, which is how a `text` search is sorted and
    the only way it is. A session is pinned and unpinned by updating it.

    A backend gets its customer's sessions; an end user gets their own, whatever the filter says: an
    `$or` chooses among the sessions the caller may see and never adds to them. An anonymous caller who
    named nobody gets none.

    The search reads what a person named the conversation, not what was said in it. Pages are followed
    by `next_cursor`, sent back with the same filter and sort. There is no total: counting every
    conversation costs more than the page.

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
