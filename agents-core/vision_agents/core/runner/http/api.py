import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import Response
from vision_agents.core import AgentLauncher
from vision_agents.core.agents.agent_launcher import AgentSession

from .dependencies import (
    can_close_session,
    can_start_session,
    can_view_metrics,
    can_view_session,
    get_current_user,
    get_launcher,
    get_session,
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
    "/sessions",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Join call with an agent",
    description="Start a new agent and have it join the specified call.",
    dependencies=[Depends(can_start_session)],
)
async def start_session(
    request: StartSessionRequest,
    launcher: AgentLauncher = Depends(get_launcher),
    user: Any = Depends(get_current_user),
) -> StartSessionResponse:
    """Start an agent and join a call."""

    try:
        session = await launcher.start_session(
            call_id=request.call_id, call_type=request.call_type, created_by=user
        )
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
        config=session.config,
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Close the agent session and remove it from call",
    dependencies=[Depends(can_close_session)],
)
async def close_session(
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> Response:
    """
    Stop an agent and remove it from a call.
    """

    closed = await launcher.close_session(session_id)
    if not closed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    return Response(status_code=204)


@router.post(
    "/sessions/{session_id}/close",
    summary="Close the agent session via sendBeacon (POST alternative to DELETE).",
    description="Alternative endpoint for agent leave via sendBeacon. "
    "sendBeacon only supports POST requests.",
    dependencies=[Depends(can_close_session)],
)
async def close_session_beacon(
    session_id: str,
    launcher: AgentLauncher = Depends(get_launcher),
) -> Response:
    """
    Stop an agent via sendBeacon (POST alternative to DELETE).
    """

    closed = await launcher.close_session(session_id)
    if not closed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    return Response(status_code=200)


@router.get(
    "/sessions/{session_id}",
    response_model=GetAgentSessionResponse,
    summary="Get info about a running agent session",
    dependencies=[Depends(can_view_session)],
)
async def get_session_info(
    session_id: str,
    session: Optional[AgentSession] = Depends(get_session),
) -> GetAgentSessionResponse:
    """
    Get info about a running agent session.
    """

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    response = GetAgentSessionResponse(
        session_id=session.id,
        call_id=session.call_id,
        config=session.config,
        session_started_at=session.started_at,
    )
    return response


@router.get(
    "/sessions/{session_id}/metrics",
    response_model=GetAgentSessionMetricsResponse,
    summary="Get info about a running agent session",
    dependencies=[Depends(can_view_metrics)],
)
async def get_session_metrics(
    session_id: str,
    session: Optional[AgentSession] = Depends(get_session),
) -> GetAgentSessionMetricsResponse:
    """
    Get info about a running agent session.
    """

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        )

    metrics_dict = session.agent.metrics.to_dict(
        fields=[
            "llm_latency_ms__avg",
            "llm_time_to_first_token_ms__avg",
            "llm_input_tokens__total",
            "llm_output_tokens__total",
            "stt_latency_ms__avg",
            "tts_latency_ms__avg",
            "realtime_audio_input_duration_ms__total",
            "realtime_audio_output_duration_ms__total",
        ]
    )
    response = GetAgentSessionMetricsResponse(
        session_id=session.id,
        call_id=session.call_id,
        session_started_at=session.started_at,
        metrics_generated_at=datetime.now(timezone.utc),
        metrics=metrics_dict,
    )
    return response


@router.get("/health")
async def health() -> Response:
    """
    Check if the server is alive.
    """
    return Response(status_code=200)


@router.get("/ready")
async def ready(launcher: AgentLauncher = Depends(get_launcher)) -> Response:
    """
    Check if the server is ready to spawn new agents.
    """
    if launcher.ready:
        return Response(status_code=200)
    else:
        raise HTTPException(
            status_code=400, detail="Server is not ready to accept requests"
        )
