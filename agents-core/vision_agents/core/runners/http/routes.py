import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from vision_agents.core import AgentLauncher
from vision_agents.core.agents.agent_launcher import AgentNotFoundError

from .models import (
    JoinCallRequest,
    JoinCallResponse,
    LeaveCallRequest,
    LeaveCallResponse,
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
    "/agents/join",
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
        agent_id = await launcher.join(
            call_id=request.call_id, call_type=request.call_type
        )

        return JoinCallResponse(
            agent_id=agent_id,
            message="Agent joining call",
        )

    except ValueError as e:
        # Duplicate call_id - return 400
        logger.warning(f"Duplicate agent request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except Exception as e:
        logger.exception("Failed to start agent")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}",
        ) from e


@router.delete(
    "/agents/{agent_id}",
    response_model=LeaveCallResponse,
    summary="Remove agent from call",
    description="Stop an agent and remove it from its current call.",
)
async def leave_call(
    agent_id: str,
    request: LeaveCallRequest,
    launcher: AgentLauncher = AgentLauncherDependency,
) -> LeaveCallResponse:
    """
    Stop an agent and remove it from a call.
    """

    try:
        await launcher.close_agent(agent_id)
    except AgentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent for call '{agent_id}' not found",
        ) from exc

    return LeaveCallResponse(agent_id=agent_id, message="Agent removed from call")


@router.post(
    "/agents/{agent_id}/leave",
    response_model=LeaveCallResponse,
    summary="Remove agent from call (sendBeacon)",
    description="Alternative endpoint for agent leave via sendBeacon. "
    "sendBeacon only supports POST requests.",
)
async def leave_call_beacon(
    agent_id: str,
    request: LeaveCallRequest,
    launcher: AgentLauncher = AgentLauncherDependency,
) -> LeaveCallResponse:
    """
    Stop an agent via sendBeacon (POST alternative to DELETE).
    """

    try:
        await launcher.close_agent(agent_id)
    except AgentNotFoundError:
        # For beacon requests, we return success even if not found
        # since the agent may have already been cleaned up
        logger.warning(f"Beacon leave: agent for call '{agent_id}' not found")

    return LeaveCallResponse(agent_id=agent_id, message="Agent removed from call")


@router.get("/alive")
async def alive() -> Response:
    """
    Check if the server is alive.
    """
    return Response(200)


@router.get("/ready")
async def ready(launcher: AgentLauncher = Depends(_get_launcher)) -> Response:
    """
    Check if the server is ready to spawn new agents.
    """
    if launcher.warmed_up:
        return Response(200)
    else:
        return JSONResponse({"detail": "Server is warming up"}, status_code=400)
