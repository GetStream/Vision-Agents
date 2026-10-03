"""RCS: Google RCS Business Messaging webhooks in, `agentMessages` bodies out."""

import base64
import binascii
import json

from ._payload import integer, iso_time, number, obj, string, with_links
from .message import AttachmentKind, Channel, OmniAttachment, OmniMessage


def parse(payload: dict[str, object]) -> list[OmniMessage]:
    """The messages in an RBM webhook body.

    Events such as `READ` and `IS_TYPING` are left out.

    Args:
        payload: The webhook body: a Pub/Sub push, whose `message.data` is the user
            message in base64, or the user message itself.

    Returns:
        The message it carries, or none.

    Raises:
        ValueError: `message.data` is not base64 encoded JSON.
    """
    data = string(obj(payload.get("message")).get("data"))
    if data:
        try:
            payload = obj(json.loads(base64.b64decode(data, validate=True)))
        except binascii.Error as exc:
            raise ValueError("message.data is not base64") from exc
    sender_id = string(payload.get("senderPhoneNumber"))
    if not sender_id or payload.get("eventType"):
        return []
    text = string(payload.get("text")) or string(
        obj(payload.get("suggestionResponse")).get("text")
    )
    attachments = []
    file = obj(obj(payload.get("userFile")).get("payload"))
    if file:
        mime_type = string(file.get("mimeType"))
        attachments.append(
            OmniAttachment(
                kind=AttachmentKind.of(mime_type),
                url=string(file.get("fileUri")),
                mime_type=mime_type,
                name=string(file.get("fileName")),
                size=integer(file.get("fileSizeBytes")),
            )
        )
    location = obj(payload.get("location"))
    if location:
        attachments.append(
            OmniAttachment(
                kind=AttachmentKind.LOCATION,
                latitude=number(location.get("latitude")),
                longitude=number(location.get("longitude")),
            )
        )
    return [
        OmniMessage(
            channel=Channel.RCS,
            conversation_id=sender_id,
            text=text,
            attachments=attachments,
            id=string(payload.get("messageId")),
            sender_id=sender_id,
            account_id=string(payload.get("agentId")),
            sent_at=iso_time(payload.get("sendTime")),
        )
    ]


def render(message: OmniMessage) -> list[dict[str, object]]:
    """The `phones/{conversation_id}/agentMessages` bodies that send a message.

    An RBM message is text or one file, so each file is a message of its own. Places are
    map links in the text. Each send needs a `messageId` of its own in its query.

    Args:
        message: The message, its `conversation_id` the person's phone number.

    Returns:
        The bodies, in the order to send them.
    """
    files = [
        attachment
        for attachment in message.attachments
        if attachment.kind is not AttachmentKind.LOCATION and attachment.url
    ]
    places = [
        attachment.link
        for attachment in message.attachments
        if attachment.kind is AttachmentKind.LOCATION
    ]
    contents: list[dict[str, object]] = []
    text = with_links(message.text, places)
    if text:
        contents.append({"text": text})
    contents.extend({"contentInfo": {"fileUrl": file.url}} for file in files)
    return [{"contentMessage": content} for content in contents]
