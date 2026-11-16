"""SMS and MMS: Telnyx Messaging webhooks in, `/v2/messages` bodies out."""

from collections.abc import Mapping

from .._payload import iso_time, integer, obj, objs, string, with_links
from ..message import AttachmentKind, Channel, OmniAttachment, OmniMessage
from ..provider import OmniProvider, Provider

# Telnyx takes at most 10 media URLs on a message.
MAX_MEDIA = 10


class TelnyxProvider(OmniProvider):
    """SMS and MMS, through Telnyx Messaging."""

    name = Provider.TELNYX
    channels = frozenset({Channel.SMS})

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The message in a Telnyx `message.received` webhook body.

        Delivery reports for messages sent (`message.sent`, `message.finalized`) are
        left out.
        """
        data = obj(payload.get("data"))
        if string(data.get("event_type")) != "message.received":
            return []
        message = obj(data.get("payload"))
        sender_id = string(obj(message.get("from")).get("phone_number"))
        recipients = objs(message.get("to"))
        return [
            OmniMessage(
                channel=Channel.SMS,
                provider=self.name,
                conversation_id=sender_id,
                text=string(message.get("text")),
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.of(string(media.get("content_type"))),
                        url=string(media.get("url")),
                        mime_type=string(media.get("content_type")),
                        size=integer(media.get("size")),
                    )
                    for media in objs(message.get("media"))
                ],
                id=string(message.get("id")),
                sender_id=sender_id,
                account_id=string(recipients[0].get("phone_number"))
                if recipients
                else "",
                sent_at=iso_time(message.get("received_at")),
            )
        ]

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The JSON bodies of the `/v2/messages` requests that send a message.

        Files go as `media_urls`, up to `MAX_MEDIA` a message, which makes it an MMS, and
        places as map links in the text. The first body carries the text. Without an
        `account_id` to send from, add a `messaging_profile_id` to each body.
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
            body: dict[str, object] = {"to": message.conversation_id}
            if message.account_id:
                body["from"] = message.account_id
            if index == 0 and text:
                body["text"] = text
            if batch:
                body["media_urls"] = batch
            bodies.append(body)
        return bodies
