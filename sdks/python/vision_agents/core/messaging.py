"""What an agent is given when somebody writes to it.

A channel outlives the conversation that filled it, so a message posted to one is a way to
reach an agent long after the call it came from ended. This is the shape of that message, in
the core rather than in a plugin so an agent can be started by one without importing
anything optional.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class InboundMessage:
    """Somebody has written to an agent that is not running.

    A message to an agent that *is* running never reaches here: the backend answers it from
    the session itself, because that agent already knows what has been said.

    Attributes:
        channel_id: The channel it was written in, which is also the agent id a session
            started to answer it should be given so its replies land back here.
        channel_type: The type of that channel.
        config_id: The agent config the last conversation on this channel ran under. Empty
            when that conversation spelled its whole spec out instead of naming one.
        text: What was written.
        message_id: The message itself, for replying in its thread rather than after it.
        user_id: Who wrote it.
        user_name: Their name, which may be empty.
        at: When it arrived.
    """

    channel_id: str
    channel_type: str = "agent"
    config_id: str = ""
    text: str = ""
    message_id: str = ""
    user_id: str = ""
    user_name: str = ""
    at: Optional[datetime] = None

    @property
    def agent_id(self) -> str:
        """The agent this was written to, which is what names the channel."""
        return self.channel_id


MessageContext = InboundMessage
