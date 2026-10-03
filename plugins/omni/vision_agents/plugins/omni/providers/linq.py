"""Linq: v3 webhooks in, `/v3/chats/{chat_id}/messages` bodies out."""

from collections.abc import Mapping

from .._payload import integer, iso_time, obj, objs, string, with_links
from ..message import AttachmentKind, Channel, OmniAttachment, OmniMessage
from ..provider import OmniProvider, Provider

_SERVICE_CHANNELS = {
    "iMessage": Channel.IMESSAGE,
    "RCS": Channel.RCS,
    "SMS": Channel.SMS,
}


class LinqProvider(OmniProvider):
    """iMessage, RCS and SMS, through Linq.

    Linq picks the service for each message, iMessage first, and Apple can downgrade a
    single message, so one Linq chat can carry all three.
    """

    name = Provider.LINQ
    channels = frozenset(_SERVICE_CHANNELS.values())

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The message in a Linq webhook body.

        Reads the `2026-02-03` webhook version, which a subscription gets with
        `?version=2026-02-03` on its URL. The channel is the message's `service`.
        Events other than `message.received` are left out.
        """
        data = obj(payload.get("data"))
        if (
            payload.get("event_type") != "message.received"
            or data.get("direction") != "inbound"
        ):
            return []
        texts = []
        attachments = []
        for part in objs(data.get("parts")):
            kind = part.get("type")
            if kind in ("text", "link"):
                texts.append(string(part.get("value")))
            elif kind == "media":
                mime_type = string(part.get("mime_type"))
                attachments.append(
                    OmniAttachment(
                        kind=AttachmentKind.of(mime_type),
                        url=string(part.get("url")),
                        media_id=string(part.get("id")),
                        mime_type=mime_type,
                        name=string(part.get("filename")),
                        size=integer(part.get("size_bytes")),
                    )
                )
        chat = obj(data.get("chat"))
        return [
            OmniMessage(
                channel=_SERVICE_CHANNELS.get(
                    string(data.get("service")), Channel.IMESSAGE
                ),
                provider=self.name,
                conversation_id=string(chat.get("id")),
                text="\n".join(text for text in texts if text),
                attachments=attachments,
                id=string(data.get("id")),
                sender_id=string(obj(data.get("sender_handle")).get("handle")),
                account_id=string(obj(chat.get("owner_handle")).get("handle")),
                reply_to=string(obj(data.get("reply_to")).get("message_id")),
                sent_at=iso_time(data.get("sent_at")),
            )
        ]

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The body that sends a message to its Linq chat.

        The text is one part and each file with a URL a media part after it. Places are
        map links in the text. Linq picks the service, as it does for any send.
        """
        places = [
            attachment.link
            for attachment in message.attachments
            if attachment.kind is AttachmentKind.LOCATION
        ]
        parts: list[dict[str, object]] = []
        text = with_links(message.text, places)
        if text:
            parts.append({"type": "text", "value": text})
        parts.extend(
            {"type": "media", "url": attachment.url}
            for attachment in message.attachments
            if attachment.kind is not AttachmentKind.LOCATION and attachment.url
        )
        if not parts:
            return []
        content: dict[str, object] = {"parts": parts}
        if message.reply_to:
            content["reply_to"] = {"message_id": message.reply_to}
        return [{"message": content}]
