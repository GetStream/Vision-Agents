from fastapi import Request
from vision_agents.core import AgentLauncher

from .options import ServeOptions


def can_start_session(call_id: str): ...


def can_close_session(call_id: str): ...


def can_view_session(call_id: str): ...


def can_view_metrics(call_id: str): ...


def get_launcher(request: Request) -> AgentLauncher:
    """Get an agent launcher from the FastAPI app."""
    return request.app.state.launcher


def get_options(request: Request) -> ServeOptions:
    return request.app.state.options
