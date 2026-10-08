"""SMS and MMS: Twilio Messaging webhooks in, `Messages.json` form bodies out."""

from collections.abc import Mapping

from .._payload import integer, string, with_links
from ..message import AttachmentKind, Channel, OmniAttachment, OmniMessage
from ..provider import OmniProvider, Provider

# Twilio takes at most 10 media URLs on a message.
MAX_MEDIA = 10

# Twilio writes WhatsApp and RCS addresses with these prefixes and SMS numbers bare.
_PREFIXED_CHANNELS = {"whatsapp:": Channel.WHATSAPP, "rcs:": Channel.RCS}


class TwilioProvider(OmniProvider):
    """SMS and MMS, through Twilio Programmable Messaging.

    Twilio WhatsApp and RCS senders use the same webhook and API, with their addresses
    prefixed by `whatsapp:` or `rcs:`, so it carries those channels too.
    """

    name = Provider.TWILIO
    channels = frozenset({Channel.SMS, Channel.WHATSAPP, Channel.RCS})

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The message in a Twilio incoming message webhook's form fields.

        Status callbacks are left out.
        """
        message_id = string(payload.get("MessageSid"))
        if not message_id or "MessageStatus" in payload:
            return []
        attachments = []
        for index in range(integer(payload.get("NumMedia")) or 0):
            mime_type = string(payload.get(f"MediaContentType{index}"))
            attachments.append(
                OmniAttachment(
                    kind=AttachmentKind.of(mime_type),
                    url=string(payload.get(f"MediaUrl{index}")),
                    mime_type=mime_type,
                )
            )
        sender_id = string(payload.get("From"))
        return [
            OmniMessage(
                channel=next(
                    (
                        channel
                        for prefix, channel in _PREFIXED_CHANNELS.items()
                        if sender_id.startswith(prefix)
                    ),
                    Channel.SMS,
                ),
                provider=self.name,
                conversation_id=sender_id,
                text=string(payload.get("Body")),
                attachments=attachments,
                id=message_id,
                sender_id=sender_id,
                account_id=string(payload.get("To")),
            )
        ]

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The form bodies of the `Messages.json` requests that send a message.

        Files go as `MediaUrl`, up to `MAX_MEDIA` a message, and places as map links in
        the text. The first body carries the text. Without an `account_id` to send from,
        add a `MessagingServiceSid` to each body.
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
