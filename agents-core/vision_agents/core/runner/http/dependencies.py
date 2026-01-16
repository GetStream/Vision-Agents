from typing import Any, Optional

from fastapi import Depends, Request
from vision_agents.core import AgentLauncher, AgentSession

from .options import ServeOptions


def get_current_user() -> Any:
    return None


def can_start_session(): ...


def can_close_session(): ...


def can_view_session(): ...


def can_view_metrics(): ...


def get_launcher(request: Request) -> AgentLauncher:
    """
    Get an agent launcher from the FastAPI app
    """
    return request.app.state.launcher


def get_options(request: Request) -> ServeOptions:
    return request.app.state.options


def get_session(
    session_id: str, launcher: AgentLauncher = Depends(get_launcher)
) -> Optional[AgentSession]:
    return launcher.get_session(session_id=session_id)
