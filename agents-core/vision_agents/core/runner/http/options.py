from typing import Any, Callable, Iterable, Optional

import pydantic_settings
from fastapi import FastAPI


def allow_all() -> Any:
    return None


class ServeOptions(pydantic_settings.BaseSettings):
    """A collection of configuration options for the "serve" mode.

    Args:
        fast_api: an optional instance of FastAPI.
            When passed, it is assumed that this is a fully-configured FastAPI instance, and no
            other configuration will be applied to it.
            Use it to completely override the API with your own.

        cors_allow_origins: CORS allow origins.
        cors_allow_methods: CORS allow methods.
        cors_allow_headers: CORS allow headers.
        cors_allow_credentials: CORS allow credentials.

        can_start_session: a callable to verify if the user can start a new session.
            Receives ``call_id`` from the URL path. Can request additional
            FastAPI dependencies via Depends().
        can_close_session: a callable to verify if the user can close a given session.
            Receives ``call_id`` from the URL path. Can request additional
            FastAPI dependencies via Depends().
        can_view_session: a callable to verify if the user can view a session.
            Receives ``call_id`` from the URL path. Can request additional
            FastAPI dependencies via Depends().
        can_view_metrics: a callable to verify if the user can view metrics for the session.
            Receives ``call_id`` from the URL path. Can request additional
            FastAPI dependencies via Depends().
    """

    fast_api: Optional[FastAPI] = None
    cors_allow_origins: Iterable[str] = ("*",)
    cors_allow_methods: Iterable[str] = ("*",)
    cors_allow_headers: Iterable[str] = ("*",)
    cors_allow_credentials: bool = True
    can_start_session: Callable = allow_all
    can_close_session: Callable = allow_all
    can_view_session: Callable = allow_all
    can_view_metrics: Callable = allow_all
