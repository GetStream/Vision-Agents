"""Slack: Events API requests in, `chat.postMessage` bodies out."""

from ._payload import integer, obj, objs, string, unix_time, with_links
from .message import AttachmentKind, Channel, OmniAttachment, OmniMessage

_MESSAGE_EVENTS = ("message", "app_mention")
# A message with a file is the only subtype a person writes. The rest are edits,
# deletions, joins and bots.
_PERSON_SUBTYPES = ("", "file_share")


def parse(payload: dict[str, object]) -> list[OmniMessage]:
    """The messages in an Events API request body.

    Messages from bots, the agent's own among them, are left out. A mention of the app
    arrives both as `message` and `app_mention` when the app subscribes to both.

    Args:
        payload: The request body.

    Returns:
        The message it carries, or none for any other event.
    """
    event = obj(payload.get("event"))
    if (
        payload.get("type") != "event_callback"
        or event.get("type") not in _MESSAGE_EVENTS
        or string(event.get("subtype")) not in _PERSON_SUBTYPES
        or event.get("bot_id")
    ):
        return []
    ts = string(event.get("ts"))
    return [
        OmniMessage(
            channel=Channel.SLACK,
            conversation_id=string(event.get("channel")),
            text=string(event.get("text")),
            attachments=[_attachment(file) for file in objs(event.get("files"))],
            id=ts,
            sender_id=string(event.get("user")),
            account_id=string(payload.get("team_id")),
            thread_id=string(event.get("thread_ts")),
            sent_at=unix_time(ts),
        )
    ]


def render(message: OmniMessage) -> list[dict[str, object]]:
    """The `chat.postMessage` bodies that send a message.

    Images with a URL are image blocks. Other files and places are links in the text.
    Attachments with no URL are left out.

    Args:
        message: The message, its `conversation_id` the Slack channel.

    Returns:
        One body, or none for a message with nothing to send.
    """
    images = [
        attachment
        for attachment in message.attachments
        if attachment.kind is AttachmentKind.IMAGE and attachment.url
    ]
    links = [
        f"<{attachment.link}|{attachment.name}>" if attachment.name else attachment.link
        for attachment in message.attachments
        if attachment.kind is not AttachmentKind.IMAGE and attachment.link
    ]
    text = with_links(message.text, links)
    if not text and not images:
        return []
    body: dict[str, object] = {"channel": message.conversation_id, "text": text}
    if message.thread_id:
        body["thread_ts"] = message.thread_id
    if images:
        blocks: list[dict[str, object]] = []
        if text:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
        blocks.extend(
            {
                "type": "image",
                "image_url": image.url,
                "alt_text": image.name or "image",
            }
            for image in images
        )
        body["blocks"] = blocks
    return [body]


def _attachment(file: dict[str, object]) -> OmniAttachment:
    mime_type = string(file.get("mimetype"))
    return OmniAttachment(
        kind=AttachmentKind.of(mime_type),
        url=string(file.get("url_private")),
        media_id=string(file.get("id")),
        mime_type=mime_type,
        name=string(file.get("name")),
        size=integer(file.get("size")),
    )
