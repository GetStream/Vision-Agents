"""One shape for a message, whichever channel it came in on or goes out to."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Channel(str, Enum):
    """Where a message is written and read."""

    SLACK = "slack"
    TEAMS = "teams"
    WHATSAPP = "whatsapp"
    RCS = "rcs"
    SMS = "sms"
    IMESSAGE = "imessage"


class AttachmentKind(str, Enum):
    """What an attachment is. All but `LOCATION` are Stream's own attachment types."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    LOCATION = "location"

    @classmethod
    def of(cls, mime_type: str) -> "AttachmentKind":
        """The kind of a file with this MIME type."""
        prefix = mime_type.partition("/")[0].lower()
        if prefix in ("image", "video", "audio"):
            return cls(prefix)
        return cls.FILE


@dataclass
class OmniAttachment:
    """A file or a place sent with a message.

    Attributes:
        kind: What it is.
        url: Where to fetch it. Slack, Twilio and RCS URLs need the account's credentials.
        media_id: The provider's id for it. WhatsApp media has only this until it is looked
            up with the account's token.
        mime_type: Its MIME type.
        name: Its file name, or the name of a place.
        size: Its size in bytes.
        latitude: A location's latitude.
        longitude: A location's longitude.
    """

    kind: AttachmentKind
    url: str = ""
    media_id: str = ""
    mime_type: str = ""
    name: str = ""
    size: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @property
    def link(self) -> str:
        """A URL for it that a person can open: a map for a location."""
        if self.kind is AttachmentKind.LOCATION:
            return f"https://maps.google.com/?q={self.latitude},{self.longitude}"
        return self.url


@dataclass
class OmniMessage:
    """A message on one of the channels.

    Attributes:
        channel: The channel it is on.
        provider: The name of the provider that carries it, a `Provider` for the built-in
            ones.
        conversation_id: Where replies go: the Slack channel, the WhatsApp, RCS or SMS
            number of the person, or the Linq chat.
        text: What was written.
        attachments: Files and places sent with it.
        id: The provider's id for it. On Slack, its `ts`.
        sender_id: Who wrote it: a Slack user, a phone number or a Linq handle.
        sender_name: Their name, when the channel tells.
        account_id: The business side of the conversation: the Slack team, the WhatsApp
            phone number id, the RCS agent, the SMS number or the Linq line.
        thread_id: The Slack thread it is in. Replies stay in it.
        reply_to: The provider's id of the message it quotes.
        sent_at: When it was sent.
    """

    channel: Channel
    provider: str
    conversation_id: str
    text: str = ""
    attachments: list[OmniAttachment] = field(default_factory=list)
    id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    account_id: str = ""
    thread_id: str = ""
    reply_to: str = ""
    sent_at: Optional[datetime] = None
