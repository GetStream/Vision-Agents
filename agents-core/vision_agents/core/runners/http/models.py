"""Pydantic models for agent API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration options for the agent."""

    llm_provider: str = Field(
        default="gemini",
        description="LLM provider to use (openai, gemini, anthropic)",
    )
    llm_model: str | None = Field(
        default=None,
        description="Specific model for the LLM provider (e.g., gpt-4o, claude-3.5-sonnet)",
    )
    stt_provider: str = Field(
        default="deepgram",
        description="Speech-to-text provider (deepgram, whisper, assembly)",
    )
    tts_provider: str = Field(
        default="cartesia",
        description="Text-to-speech provider (cartesia, elevenlabs, openai)",
    )
    tts_voice: str | None = Field(
        default=None,
        description="Voice ID for the TTS provider",
    )
    turn_detection: str = Field(
        default="silero_vad",
        description="Turn detection mode (silero_vad, semantic)",
    )
    instructions: str = Field(
        default="You're a voice AI assistant. Keep responses short and conversational. Don't use special characters or formatting. Be friendly and helpful.",
        description="System instructions for the agent",
    )
    agent_name: str = Field(
        default="agent",
        description="Display name for the agent in the call",
    )
    agent_id: str = Field(
        default="agent",
        description="Unique identifier for the agent user",
    )


class UpdateAgentConfig(BaseModel):
    """Configuration fields that can be updated (all optional for partial updates)."""

    llm_provider: str | None = Field(
        default=None,
        description="LLM provider to use (openai, gemini, anthropic)",
    )
    llm_model: str | None = Field(
        default=None,
        description="Specific model for the LLM provider (e.g., gpt-4o, claude-3.5-sonnet)",
    )
    stt_provider: str | None = Field(
        default=None,
        description="Speech-to-text provider (deepgram, whisper, assembly)",
    )
    tts_provider: str | None = Field(
        default=None,
        description="Text-to-speech provider (cartesia, elevenlabs, openai)",
    )
    tts_voice: str | None = Field(
        default=None,
        description="Voice ID for the TTS provider",
    )
    turn_detection: str | None = Field(
        default=None,
        description="Turn detection mode (silero_vad, semantic)",
    )
    instructions: str | None = Field(
        default=None,
        description="System instructions for the agent",
    )


class UpdateAgentRequest(BaseModel):
    """Request body for updating an agent configuration."""

    user_id: str = Field(..., description="ID of the user requesting the update")
    config: UpdateAgentConfig = Field(..., description="Configuration fields to update")


class UpdateAgentResponse(BaseModel):
    """Response after updating an agent."""

    agent_id: str = Field(..., description="The agent ID")
    message: str = Field(..., description="Status message")


class JoinCallRequest(BaseModel):
    """Request body for joining a call."""

    call_id: str = Field(..., description="Unique identifier of the call to join")
    call_type: str = Field(default="default", description="Type of the call to join")
    user_id: str = Field(..., description="ID of the user requesting the agent")
    config: AgentConfig | None = Field(
        default=None,
        description="Optional agent configuration. Uses defaults if not provided.",
    )


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


class AgentSessionInfo(BaseModel):
    """Information about an active agent session."""

    call_id: str
    user_id: str
    config: AgentConfig
    created_at: datetime
    last_seen: datetime
    is_running: bool


class ListAgentsResponse(BaseModel):
    """Response containing all active agent sessions."""

    agents: list[AgentSessionInfo]
    count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
