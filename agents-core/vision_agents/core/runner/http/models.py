"""Pydantic models for agent API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class JoinCallRequest(BaseModel):
    """Request body for joining a call."""

    call_id: str = Field(..., description="Unique identifier of the call to join")
    call_type: str = Field(default="default", description="Type of the call to join")
    user_id: str = Field(..., description="ID of the user requesting the agent")


class JoinCallResponse(BaseModel):
    """Response after successfully starting an agent."""

    session_id: str = Field(..., description="The ID of the agent session")
    call_id: str = Field(..., description="The ID of the call")
    config: dict  # TODO: Make it a type
    session_started_at: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


class GetAgentSessionResponse(BaseModel):
    """Information about an active agent session."""

    session_id: str
    call_id: str
    config: dict  # TODO: Make it a type
    session_started_at: datetime


class GetAgentSessionMetricsResponse(BaseModel):
    """Information about an active agent session."""

    session_id: str = Field(..., description="The ID of the agent session")
    call_id: str = Field(..., description="The ID of the current call")
    metrics: dict[str, int | float | None] = Field(
        ..., description="Dictionary of metrics"
    )
    session_started_at: datetime = Field(
        ..., description="Date and time in UTC at which the session was started"
    )
    metrics_generated_at: datetime = Field(
        ..., description="Date and time in UTC at which the metrics were generated"
    )
