import datetime
from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.spend_scope import SpendScope
from ...models.usage_event_page import UsageEventPage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    scope: SpendScope | Unset = UNSET,
    app_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    meter: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    limit: int | Unset = 50,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_scope: str | Unset = UNSET
    if not isinstance(scope, Unset):
        json_scope = scope.value

    params["scope"] = json_scope

    params["app_id"] = app_id

    json_from_: str | Unset = UNSET
    if not isinstance(from_, Unset):
        json_from_ = from_.isoformat()
    params["from"] = json_from_

    json_to: str | Unset = UNSET
    if not isinstance(to, Unset):
        json_to = to.isoformat()
    params["to"] = json_to

    params["meter"] = meter

    params["api_key"] = api_key

    params["user_id"] = user_id

    params["team"] = team

    params["category"] = category

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/usage/events",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | UsageEventPage | None:
    if response.status_code == 200:
        response_200 = UsageEventPage.from_dict(response.json())

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
) -> Response[ErrorResponse | UsageEventPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    scope: SpendScope | Unset = UNSET,
    app_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    meter: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    limit: int | Unset = 50,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | UsageEventPage]:
    """List the usage events

     Every unit of work recorded for the calling app, or with scope=organization for every app of its
    organization (app_id narrows that to one), newest first: by when it happened, then by usage_id. Each
    names who it was for, what it was, how much of it there was and what the organization is charged.
    Narrow it by from and to, meter, api_key, user_id, and team or category; send the filters again with
    cursor for the next page. Work reaches the list once the router has shipped it to the usage service,
    so the newest may take a moment to appear. With no usage service, or for an app in no organization,
    it answers 404 billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        scope (SpendScope | Unset): Whose spend is counted, or whose usage events are listed: the
            calling app's, or every app's of its organization. organization is server-side only.
        app_id (str | Unset): With scope=organization, only this app's events. With scope=app the
            app is the caller's, and another is refused.
        from_ (datetime.datetime | Unset): Only events that happened at or after this, inclusive.
        to (datetime.datetime | Unset): Only events that happened before this, exclusive.
        meter (str | Unset): Only this kind of work, such as llm or runtime.
        api_key (str | Unset): Only work started with this public API key, or by this origin of
            work no key started, such as hook:chat.
        user_id (str | Unset): Only work for this end user of the app. With scope=organization it
            needs app_id, since a user id is only unique within its app.
        team (str | Unset): Only work for this team: its users' tokens name it in their team
            claim. Not with category.
        category (str | Unset): Only work carrying this usage category, as the tokens name it. Not
            with team.
        limit (int | Unset): The most events one page holds. Default: 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UsageEventPage]
    """

    kwargs = _get_kwargs(
        scope=scope,
        app_id=app_id,
        from_=from_,
        to=to,
        meter=meter,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    scope: SpendScope | Unset = UNSET,
    app_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    meter: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    limit: int | Unset = 50,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | UsageEventPage | None:
    """List the usage events

     Every unit of work recorded for the calling app, or with scope=organization for every app of its
    organization (app_id narrows that to one), newest first: by when it happened, then by usage_id. Each
    names who it was for, what it was, how much of it there was and what the organization is charged.
    Narrow it by from and to, meter, api_key, user_id, and team or category; send the filters again with
    cursor for the next page. Work reaches the list once the router has shipped it to the usage service,
    so the newest may take a moment to appear. With no usage service, or for an app in no organization,
    it answers 404 billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        scope (SpendScope | Unset): Whose spend is counted, or whose usage events are listed: the
            calling app's, or every app's of its organization. organization is server-side only.
        app_id (str | Unset): With scope=organization, only this app's events. With scope=app the
            app is the caller's, and another is refused.
        from_ (datetime.datetime | Unset): Only events that happened at or after this, inclusive.
        to (datetime.datetime | Unset): Only events that happened before this, exclusive.
        meter (str | Unset): Only this kind of work, such as llm or runtime.
        api_key (str | Unset): Only work started with this public API key, or by this origin of
            work no key started, such as hook:chat.
        user_id (str | Unset): Only work for this end user of the app. With scope=organization it
            needs app_id, since a user id is only unique within its app.
        team (str | Unset): Only work for this team: its users' tokens name it in their team
            claim. Not with category.
        category (str | Unset): Only work carrying this usage category, as the tokens name it. Not
            with team.
        limit (int | Unset): The most events one page holds. Default: 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UsageEventPage
    """

    return sync_detailed(
        client=client,
        scope=scope,
        app_id=app_id,
        from_=from_,
        to=to,
        meter=meter,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    scope: SpendScope | Unset = UNSET,
    app_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    meter: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    limit: int | Unset = 50,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | UsageEventPage]:
    """List the usage events

     Every unit of work recorded for the calling app, or with scope=organization for every app of its
    organization (app_id narrows that to one), newest first: by when it happened, then by usage_id. Each
    names who it was for, what it was, how much of it there was and what the organization is charged.
    Narrow it by from and to, meter, api_key, user_id, and team or category; send the filters again with
    cursor for the next page. Work reaches the list once the router has shipped it to the usage service,
    so the newest may take a moment to appear. With no usage service, or for an app in no organization,
    it answers 404 billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        scope (SpendScope | Unset): Whose spend is counted, or whose usage events are listed: the
            calling app's, or every app's of its organization. organization is server-side only.
        app_id (str | Unset): With scope=organization, only this app's events. With scope=app the
            app is the caller's, and another is refused.
        from_ (datetime.datetime | Unset): Only events that happened at or after this, inclusive.
        to (datetime.datetime | Unset): Only events that happened before this, exclusive.
        meter (str | Unset): Only this kind of work, such as llm or runtime.
        api_key (str | Unset): Only work started with this public API key, or by this origin of
            work no key started, such as hook:chat.
        user_id (str | Unset): Only work for this end user of the app. With scope=organization it
            needs app_id, since a user id is only unique within its app.
        team (str | Unset): Only work for this team: its users' tokens name it in their team
            claim. Not with category.
        category (str | Unset): Only work carrying this usage category, as the tokens name it. Not
            with team.
        limit (int | Unset): The most events one page holds. Default: 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | UsageEventPage]
    """

    kwargs = _get_kwargs(
        scope=scope,
        app_id=app_id,
        from_=from_,
        to=to,
        meter=meter,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    scope: SpendScope | Unset = UNSET,
    app_id: str | Unset = UNSET,
    from_: datetime.datetime | Unset = UNSET,
    to: datetime.datetime | Unset = UNSET,
    meter: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    limit: int | Unset = 50,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | UsageEventPage | None:
    """List the usage events

     Every unit of work recorded for the calling app, or with scope=organization for every app of its
    organization (app_id narrows that to one), newest first: by when it happened, then by usage_id. Each
    names who it was for, what it was, how much of it there was and what the organization is charged.
    Narrow it by from and to, meter, api_key, user_id, and team or category; send the filters again with
    cursor for the next page. Work reaches the list once the router has shipped it to the usage service,
    so the newest may take a moment to appear. With no usage service, or for an app in no organization,
    it answers 404 billing_not_enabled. A suspended organization reads it all the same.

    Server-side only: a request that is not server-side answers 403 server_side_only.

    Args:
        scope (SpendScope | Unset): Whose spend is counted, or whose usage events are listed: the
            calling app's, or every app's of its organization. organization is server-side only.
        app_id (str | Unset): With scope=organization, only this app's events. With scope=app the
            app is the caller's, and another is refused.
        from_ (datetime.datetime | Unset): Only events that happened at or after this, inclusive.
        to (datetime.datetime | Unset): Only events that happened before this, exclusive.
        meter (str | Unset): Only this kind of work, such as llm or runtime.
        api_key (str | Unset): Only work started with this public API key, or by this origin of
            work no key started, such as hook:chat.
        user_id (str | Unset): Only work for this end user of the app. With scope=organization it
            needs app_id, since a user id is only unique within its app.
        team (str | Unset): Only work for this team: its users' tokens name it in their team
            claim. Not with category.
        category (str | Unset): Only work carrying this usage category, as the tokens name it. Not
            with team.
        limit (int | Unset): The most events one page holds. Default: 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | UsageEventPage
    """

    return (
        await asyncio_detailed(
            client=client,
            scope=scope,
            app_id=app_id,
            from_=from_,
            to=to,
            meter=meter,
            api_key=api_key,
            user_id=user_id,
            team=team,
            category=category,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
