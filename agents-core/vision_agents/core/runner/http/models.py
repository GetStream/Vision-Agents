"""Pydantic models for agent API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    """Request body for joining a call."""

    call_id: str = Field(..., description="Unique identifier of the call to join")
    call_type: str = Field(default="default", description="Type of the call to join")


class StartSessionResponse(BaseModel):
    """Response after successfully starting an agent."""

    session_id: str = Field(..., description="The ID of the agent session")
    call_id: str = Field(..., description="The ID of the call")
    session_started_at: datetime


class GetAgentSessionResponse(BaseModel):
    """Details about an active agent session."""

    session_id: str
    call_id: str
    session_started_at: datetime


class GetAgentSessionMetricsResponse(BaseModel):
    """Metrics of the active agent session."""

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
