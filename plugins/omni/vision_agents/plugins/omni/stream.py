"""Omni messages as Stream Chat messages, and back.

A message keeps its text as the Stream message text and its files as Stream attachments.
Everything else about it, which a reply needs to find its way back, is the message's
`custom.omni`.
"""

import hashlib
from typing import Optional

from getstream.models import Attachment, MessageRequest, MessageResponse

from ._payload import integer, iso_time, number, obj, string
from .message import AttachmentKind, Channel, OmniAttachment, OmniMessage

OMNI_KEY = "omni"
SCHEMA_VERSION = 1

# 96 bits keeps ids unique and short: Stream channel ids are at most 64 characters.
_DIGEST_LENGTH = 24


def stream_user_id(message: OmniMessage) -> str:
    """The Stream user who wrote a message, one per sender per provider.

    It is a digest, valid whatever the sender id. The sender id itself is in
    `custom.omni`.
    """
    return _stream_id(message.provider, message.sender_id)


def stream_channel_id(message: OmniMessage) -> str:
    """The Stream channel a conversation is kept in.

    There is one per provider, account and conversation, so a person writing to two of
    the business's numbers has two. A Linq chat stays one channel when a message in it
    falls back from iMessage to SMS. It is a digest, at most 64 characters whatever the
    ids; the ids themselves are in `custom.omni`.
    """
    return _stream_id(message.provider, message.account_id, message.conversation_id)


def to_stream(message: OmniMessage, user_id: Optional[str] = None) -> MessageRequest:
    """The Stream message to send for an omni message.

    Args:
        message: The message.
        user_id: The Stream user to send it as. Defaults to `stream_user_id(message)`.

    Returns:
        The request, with the message's routing in `custom.omni`.
    """
    omni: dict[str, object] = {
        "v": SCHEMA_VERSION,
        "channel": message.channel.value,
        "provider": message.provider,
        "conversation_id": message.conversation_id,
        "id": message.id,
        "sender_id": message.sender_id,
        "sender_name": message.sender_name,
        "account_id": message.account_id,
        "thread_id": message.thread_id,
        "reply_to": message.reply_to,
        "sent_at": message.sent_at.isoformat() if message.sent_at else None,
    }
    return MessageRequest(
        text=message.text,
        user_id=user_id or stream_user_id(message),
        attachments=[_to_attachment(attachment) for attachment in message.attachments],
        custom={OMNI_KEY: _compact(omni)},
    )


def from_stream(
    message: MessageRequest | MessageResponse, route: Optional[OmniMessage] = None
) -> OmniMessage:
    """The omni message a Stream message is.

    Attachments of types other than an `AttachmentKind`, such as the agent's reasoning
    steps and tool calls, are left out.

    Args:
        message: The Stream message.
        route: The message it answers. The result goes back to where that came from.
            Without it, the route is the message's own `custom.omni`, as `to_stream` wrote it.

    Returns:
        The message.

    Raises:
        ValueError: There is no route and the message has no `custom.omni`.
    """
    text = message.text or ""
    attachments = [
        attachment
        for attachment in map(_from_attachment, message.attachments or [])
        if attachment is not None
    ]
    if route is not None:
        return OmniMessage(
            channel=route.channel,
            provider=route.provider,
            conversation_id=route.conversation_id,
            account_id=route.account_id,
            thread_id=route.thread_id,
            text=text,
            attachments=attachments,
        )
    omni = obj((message.custom or {}).get(OMNI_KEY))
    if not omni:
        raise ValueError(
            f"message has no custom.{OMNI_KEY}; pass the message it answers as route"
        )
    return OmniMessage(
        channel=Channel(string(omni.get("channel"))),
        provider=string(omni.get("provider")),
        conversation_id=string(omni.get("conversation_id")),
        text=text,
        attachments=attachments,
        id=string(omni.get("id")),
        sender_id=string(omni.get("sender_id")),
        sender_name=string(omni.get("sender_name")),
        account_id=string(omni.get("account_id")),
        thread_id=string(omni.get("thread_id")),
        reply_to=string(omni.get("reply_to")),
        sent_at=iso_time(omni.get("sent_at")),
    )


def _stream_id(provider: str, *ids: str) -> str:
    digest = hashlib.sha256("\0".join(ids).encode()).hexdigest()
    return f"{provider}_{digest[:_DIGEST_LENGTH]}"


def _compact(values: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value not in (None, "")}


def _to_attachment(attachment: OmniAttachment) -> Attachment:
    custom = _compact(
        {
            "mime_type": attachment.mime_type,
            "file_size": attachment.size,
            "media_id": attachment.media_id,
            "latitude": attachment.latitude,
            "longitude": attachment.longitude,
        }
    )
    title = attachment.name or None
    if attachment.kind is AttachmentKind.IMAGE:
        return Attachment(
            type="image", image_url=attachment.url or None, title=title, custom=custom
        )
    if attachment.kind is AttachmentKind.LOCATION:
        return Attachment(
            type="location", title=title, title_link=attachment.link, custom=custom
        )
    return Attachment(
        type=attachment.kind.value,
        asset_url=attachment.url or None,
        title=title,
        custom=custom,
    )


def _from_attachment(attachment: Attachment) -> Optional[OmniAttachment]:
    if attachment.type not in {kind.value for kind in AttachmentKind}:
        return None
    custom = attachment.custom or {}
    return OmniAttachment(
        kind=AttachmentKind(attachment.type),
        url=attachment.image_url or attachment.asset_url or "",
        media_id=string(custom.get("media_id")),
        mime_type=string(custom.get("mime_type")),
        name=attachment.title or "",
        size=integer(custom.get("file_size")),
        latitude=number(custom.get("latitude")),
        longitude=number(custom.get("longitude")),
    )
