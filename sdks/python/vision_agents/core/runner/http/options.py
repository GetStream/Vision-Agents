import ipaddress
from typing import Any, Callable, Iterable, Optional

import pydantic_settings
from fastapi import FastAPI, HTTPException, Request, status

# Pages served from this machine, on any port, such as a frontend under development.
LOCAL_ORIGINS = r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?"


def allow_all() -> Any:
    return None


def allow_local(request: Request) -> None:
    """Allow a caller on this machine and refuse any other.

    The default for every permission, so a server bound beyond loopback refuses the network
    until it is given checks of its own. A reverse proxy on the same machine is a caller on
    this machine.
    """
    host = request.client.host if request.client else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if address is None or not address.is_loopback:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only a caller on this machine is allowed; pass permission checks to ServeOptions",
        )


class ServeOptions(pydantic_settings.BaseSettings):
    """A collection of configuration options for the "serve" mode.

    Args:
        fast_api: an optional instance of FastAPI.
            When passed, it is assumed that this is a fully-configured FastAPI instance, and no
            other configuration will be applied to it.
            Use it to completely override the API with your own.

        cors_allow_origins: CORS allow origins. None by default; ``["*"]`` allows every origin.
        cors_allow_origin_regex: CORS allow origins matching this regex. Pages served from
            this machine by default; ``None`` for none.
        cors_allow_methods: CORS allow methods.
        cors_allow_headers: CORS allow headers.
        cors_allow_credentials: CORS allow credentials. Off by default.

        can_start_session: a callable to verify if the user can start a new session.
            Receives ``call_id`` from the URL path. Can request additional
            FastAPI dependencies via Depends(). Every permission defaults to
            ``allow_local``, which refuses a caller on another machine; ``allow_all``
            allows everyone.
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
    cors_allow_origins: Iterable[str] = ()
    cors_allow_origin_regex: Optional[str] = LOCAL_ORIGINS
    cors_allow_methods: Iterable[str] = ("*",)
    cors_allow_headers: Iterable[str] = ("*",)
    cors_allow_credentials: bool = False
    can_start_session: Callable = allow_local
    can_close_session: Callable = allow_local
    can_view_session: Callable = allow_local
    can_view_metrics: Callable = allow_local
