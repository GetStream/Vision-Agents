import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import Response
from vision_agents.core import AgentLauncher
from vision_agents.core.agents.exceptions import SessionLimitExceeded

from .dependencies import (
    can_close_session,
    can_start_session,
    can_view_metrics,
    can_view_session,
    get_launcher,
)
from .models import (
    GetAgentSessionMetricsResponse,
    GetAgentSessionResponse,
    StartSessionRequest,
    StartSessionResponse,
)

__all__ = ["router", "lifespan"]


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    launcher: AgentLauncher = app.state.launcher

    try:
        await launcher.start()
        yield
    finally:
        await launcher.stop()


router = APIRouter()


@router.post(
    "/calls/{call_id}/sessions",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Join call with an agent",
    description="Start a new agent and have it join the specified call.",
    responses={
        201: {
            "description": "Session created successfully",
            "model": StartSessionResponse,
        },
        429: {
            "description": "Session limits exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Reached maximum concurrent sessions of X",
                    }
                }
            },
        },
    },
    dependencies=[Depends(can_start_session)],
)
async def start_session(
    call_id: str,
    request: StartSessionRequest,
    launcher: AgentLauncher = Depends(get_launcher),
) -> StartSessionResponse:
    """Start an agent and join a call."""

    try:
        session = await launcher.start_session(
            call_id=call_id, call_type=request.call_type
        )
    except SessionLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except Exception as e:
        logger.exception("Failed to start agent")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}",
        ) from e

    return StartSessionResponse(
        session_id=session.id,
        call_id=session.call_id,
        session_started_at=session.started_at,
    )


@router.delete(
    "/calls/{call_id}/sessions/{session_id}",
    summary="Request closure of an agent session",
    dependencies=[Depends(can_close_session)],
)
async def close_session(
    call_id: str,
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> Response:
    """Request closure of an agent session.

    Sets a close flag in the registry. The owning node will close the
    session on its next maintenance cycle.
    """
    info = await launcher.get_session_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )
    await launcher.request_close_session(session_id)
    return Response(status_code=202)


@router.post(
    "/calls/{call_id}/sessions/{session_id}/close",
    summary="Request closure of an agent session (sendBeacon alternative)",
    description="Alternative endpoint for requesting session closure via the "
    "browser sendBeacon API, which only supports POST requests.",
    dependencies=[Depends(can_close_session)],
)
async def close_session_beacon(
    call_id: str,
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> Response:
    """Request closure of an agent session via sendBeacon.

    Sets a close flag in the registry. The owning node will close the
    session on its next maintenance cycle.
    """
    info = await launcher.get_session_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )
    await launcher.request_close_session(session_id)
    return Response(status_code=202)


@router.get(
    "/calls/{call_id}/sessions/{session_id}",
    response_model=GetAgentSessionResponse,
    summary="Get info about a running agent session",
    dependencies=[Depends(can_view_session)],
)
async def get_session_info(
    call_id: str,
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> GetAgentSessionResponse:
    """Get info about a running agent session."""

    info = await launcher.get_session_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    return GetAgentSessionResponse(
        session_id=info.session_id,
        call_id=info.call_id,
        session_started_at=datetime.fromtimestamp(info.started_at, tz=timezone.utc),
    )


@router.get(
    "/calls/{call_id}/sessions/{session_id}/metrics",
    response_model=GetAgentSessionMetricsResponse,
    summary="Get metrics for a running agent session",
    dependencies=[Depends(can_view_metrics)],
)
async def get_session_metrics(
    call_id: str,
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> GetAgentSessionMetricsResponse:
    """Get metrics for a running agent session from the registry."""

    info = await launcher.get_session_info(session_id)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    return GetAgentSessionMetricsResponse(
        session_id=info.session_id,
        call_id=info.call_id,
        session_started_at=datetime.fromtimestamp(info.started_at, tz=timezone.utc),
        metrics_generated_at=datetime.fromtimestamp(
            info.metrics_updated_at, tz=timezone.utc
        ),
        metrics=info.metrics,
    )


@router.get("/health")
async def health() -> Response:
    """Check if the server is alive."""
    return Response(status_code=200)


@router.get("/ready")
async def ready(launcher: AgentLauncher = Depends(get_launcher)) -> Response:
    """Check if the server is ready to spawn new agents."""
    if launcher.ready:
        return Response(status_code=200)
    else:
        raise HTTPException(
            status_code=400, detail="Server is not ready to accept requests"
        )
