import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from vision_agents.core import AgentLauncher
from vision_agents.core.agents.agent_launcher import SessionNotFoundError

from .models import (
    GetAgentSessionResponse,
    JoinCallRequest,
    JoinCallResponse,
)

__all__ = ["router"]


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    launcher: AgentLauncher = app.state.launcher

    try:
        await launcher.start()
        yield
    finally:
        await launcher.stop()


router = APIRouter(lifespan=lifespan)


def _get_launcher(request: Request) -> AgentLauncher:
    """
    Get an agent launcher from the FastAPI app
    """
    return request.app.state.launcher


AgentLauncherDependency = Depends(_get_launcher)


@router.post(
    "/sessions",
    response_model=JoinCallResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Join call with an agent",
    description="Start a new agent and have it join the specified call.",
)
async def join_call(
    request: JoinCallRequest, launcher: AgentLauncher = AgentLauncherDependency
) -> JoinCallResponse:
    """Start an agent and join a call."""

    try:
        session = await launcher.start_session(
            call_id=request.call_id, call_type=request.call_type
        )
    except Exception as e:
        logger.exception("Failed to start agent")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}",
        ) from e

    return JoinCallResponse(
        session_id=session.id,
        call_id=session.call_id,
        started_at=session.started_at,
        config=session.config,
    )


@router.delete(
    "/session/{session_id}",
    summary="Close the agent session and remove it from call",
)
async def close_session(
    session_id: str,
    launcher: AgentLauncher = AgentLauncherDependency,
) -> Response:
    """
    Stop an agent and remove it from a call.
    """

    try:
        await launcher.close_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent for call '{session_id}' not found",
        ) from exc

    return Response(status_code=204)


@router.post(
    "/sessions/{session_id}/leave",
    summary="Close the agent session via sendBeacon (POST alternative to DELETE).",
    description="Alternative endpoint for agent leave via sendBeacon. "
    "sendBeacon only supports POST requests.",
)
async def close_session_beacon(
    session_id: str,
    launcher: AgentLauncher = AgentLauncherDependency,
) -> Response:
    """
    Stop an agent via sendBeacon (POST alternative to DELETE).
    """

    try:
        await launcher.close_session(session_id)
    except SessionNotFoundError:
        # For beacon requests, we return success even if not found
        # since the agent may have already been cleaned up
        logger.warning(f"Beacon leave: agent session with id '{session_id}' not found")

    return Response(status_code=200)


@router.get(
    "/sessions/{session_id}",
    response_model=GetAgentSessionResponse,
    summary="Get info about a running agent session",
)
async def get_session_info(
    session_id: str,
    launcher: AgentLauncher = AgentLauncherDependency,
) -> GetAgentSessionResponse:
    """
    Get info about a running agent session.
    """

    try:
        session = launcher.get_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session with id '{session_id}' not found",
        ) from exc

    response = GetAgentSessionResponse(
        session_id=session.id,
        call_id=session.call_id,
        config=session.config,
        started_at=session.started_at,
    )
    return response


@router.get("/health")
async def health() -> Response:
    """
    Check if the server is alive.
    """
    return Response(status_code=200)


@router.get("/ready")
async def ready(launcher: AgentLauncher = Depends(_get_launcher)) -> Response:
    """
    Check if the server is ready to spawn new agents.
    """
    if launcher.warmed_up and launcher.running:
        return Response(status_code=200)
    else:
        raise HTTPException(
            status_code=400, detail="Server is not ready to accept requests"
        )
