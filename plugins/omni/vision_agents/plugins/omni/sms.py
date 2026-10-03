"""SMS and MMS: Twilio Messaging webhooks in, `Messages.json` form bodies out."""

from collections.abc import Mapping

from ._payload import integer, with_links
from .message import AttachmentKind, Channel, OmniAttachment, OmniMessage

# Twilio takes at most 10 media URLs on a message.
MAX_MEDIA = 10


def parse(form: Mapping[str, str]) -> list[OmniMessage]:
    """The message in a Twilio incoming message webhook.

    Status callbacks are left out.

    Args:
        form: The webhook's form fields.

    Returns:
        The message it carries, or none.
    """
    if "MessageSid" not in form or "MessageStatus" in form:
        return []
    attachments = []
    for index in range(integer(form.get("NumMedia", "0")) or 0):
        mime_type = form.get(f"MediaContentType{index}", "")
        attachments.append(
            OmniAttachment(
                kind=AttachmentKind.of(mime_type),
                url=form.get(f"MediaUrl{index}", ""),
                mime_type=mime_type,
            )
        )
    sender_id = form.get("From", "")
    return [
        OmniMessage(
            channel=Channel.SMS,
            conversation_id=sender_id,
            text=form.get("Body", ""),
            attachments=attachments,
            id=form["MessageSid"],
            sender_id=sender_id,
            account_id=form.get("To", ""),
        )
    ]


def render(message: OmniMessage) -> list[dict[str, object]]:
    """The form bodies of the Twilio `Messages.json` requests that send a message.

    Files go as `MediaUrl`, up to `MAX_MEDIA` a message. Places are map links in the
    text.

    Args:
        message: The message, its `conversation_id` the person's number and its
            `account_id` the number to send from. Without one, add a
            `MessagingServiceSid` to each body.

    Returns:
        The bodies, the first one carrying the text.
    """
    urls = [
        attachment.url
        for attachment in message.attachments
        if attachment.kind is not AttachmentKind.LOCATION and attachment.url
    ]
    places = [
        attachment.link
        for attachment in message.attachments
        if attachment.kind is AttachmentKind.LOCATION
    ]
    text = with_links(message.text, places)
    batches = [
        urls[start : start + MAX_MEDIA] for start in range(0, len(urls), MAX_MEDIA)
    ]
    if text and not batches:
        batches = [[]]
    bodies: list[dict[str, object]] = []
    for index, batch in enumerate(batches):
        body: dict[str, object] = {"To": message.conversation_id}
        if message.account_id:
            body["From"] = message.account_id
        if index == 0 and text:
            body["Body"] = text
        if batch:
            body["MediaUrl"] = batch
        bodies.append(body)
    return bodies
