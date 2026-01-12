"""Pydantic models for agent API requests and responses."""

from pydantic import BaseModel, Field


class JoinCallRequest(BaseModel):
    """Request body for joining a call."""

    call_id: str = Field(..., description="Unique identifier of the call to join")
    call_type: str = Field(default="default", description="Type of the call to join")
    user_id: str = Field(..., description="ID of the user requesting the agent")


class JoinCallResponse(BaseModel):
    """Response after successfully starting an agent."""

    agent_id: str = Field(..., description="The agent ID that joined the call")
    message: str = Field(..., description="Status message")


class LeaveCallRequest(BaseModel):
    """Request body for leaving a call (used by DELETE and sendBeacon POST)."""

    user_id: str = Field(..., description="ID of the user requesting agent removal")


class LeaveCallResponse(BaseModel):
    """Response after agent leaves a call."""

    agent_id: str = Field(..., description="The agent ID that left the call")
    message: str = Field(..., description="Status message")


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
