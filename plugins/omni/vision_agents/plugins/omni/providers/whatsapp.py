"""WhatsApp: Cloud API webhooks in, `/{phone-number-id}/messages` bodies out."""

from collections.abc import Mapping
from typing import Optional

from .._payload import number, obj, objs, string, unix_time
from ..message import AttachmentKind, Channel, OmniAttachment, OmniMessage
from ..provider import OmniProvider, Provider

_MEDIA_KINDS = {
    "image": AttachmentKind.IMAGE,
    "sticker": AttachmentKind.IMAGE,
    "video": AttachmentKind.VIDEO,
    "audio": AttachmentKind.AUDIO,
    "document": AttachmentKind.FILE,
}
_MEDIA_TYPES = {
    AttachmentKind.IMAGE: "image",
    AttachmentKind.VIDEO: "video",
    AttachmentKind.AUDIO: "audio",
    AttachmentKind.FILE: "document",
}


class WhatsAppProvider(OmniProvider):
    """WhatsApp, through Meta's Cloud API."""

    name = Provider.WHATSAPP
    channels = frozenset({Channel.WHATSAPP})

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The messages in a Cloud API webhook body.

        Media arrives as an id only, in `OmniAttachment.media_id`: its URL is looked up
        with the account's token. Delivery statuses, reactions and unsupported messages
        are left out.
        """
        messages = []
        for entry in objs(payload.get("entry")):
            for change in objs(entry.get("changes")):
                value = obj(change.get("value"))
                account_id = string(obj(value.get("metadata")).get("phone_number_id"))
                names = {
                    string(contact.get("wa_id")): string(
                        obj(contact.get("profile")).get("name")
                    )
                    for contact in objs(value.get("contacts"))
                }
                for item in objs(value.get("messages")):
                    message = _message(item)
                    if message is None:
                        continue
                    message.account_id = account_id
                    message.sender_name = names.get(message.sender_id, "")
                    messages.append(message)
        return messages

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The Cloud API bodies that send a message, one per text and per attachment.

        Attachments go by URL, or by media id when they have no URL.
        """
        bodies: list[dict[str, object]] = []
        if message.text:
            bodies.append({"type": "text", "text": {"body": message.text}})
        for attachment in message.attachments:
            if attachment.kind is AttachmentKind.LOCATION:
                location: dict[str, object] = {
                    "latitude": attachment.latitude,
                    "longitude": attachment.longitude,
                }
                if attachment.name:
                    location["name"] = attachment.name
                bodies.append({"type": "location", "location": location})
                continue
            if attachment.url:
                media: dict[str, object] = {"link": attachment.url}
            elif attachment.media_id:
                media = {"id": attachment.media_id}
            else:
                continue
            if attachment.kind is AttachmentKind.FILE and attachment.name:
                media["filename"] = attachment.name
            media_type = _MEDIA_TYPES[attachment.kind]
            bodies.append({"type": media_type, media_type: media})
        for body in bodies:
            body.update(
                messaging_product="whatsapp",
                recipient_type="individual",
                to=message.conversation_id,
            )
            if message.reply_to:
                body["context"] = {"message_id": message.reply_to}
        return bodies


def _message(item: dict[str, object]) -> Optional[OmniMessage]:
    kind = string(item.get("type"))
    content = obj(item.get(kind))
    text = ""
    attachments: list[OmniAttachment] = []
    if kind == "text":
        text = string(content.get("body"))
    elif kind in _MEDIA_KINDS:
        text = string(content.get("caption"))
        attachments.append(
            OmniAttachment(
                kind=_MEDIA_KINDS[kind],
                media_id=string(content.get("id")),
                mime_type=string(content.get("mime_type")),
                name=string(content.get("filename")),
            )
        )
    elif kind == "location":
        attachments.append(
            OmniAttachment(
                kind=AttachmentKind.LOCATION,
                name=string(content.get("name")) or string(content.get("address")),
                latitude=number(content.get("latitude")),
                longitude=number(content.get("longitude")),
            )
        )
    elif kind == "button":
        text = string(content.get("text"))
    elif kind == "interactive":
        reply = obj(content.get("button_reply")) or obj(content.get("list_reply"))
        text = string(reply.get("title"))
    else:
        return None
    sender_id = string(item.get("from"))
    return OmniMessage(
        channel=Channel.WHATSAPP,
        provider=Provider.WHATSAPP,
        conversation_id=sender_id,
        text=text,
        attachments=attachments,
        id=string(item.get("id")),
        sender_id=sender_id,
        reply_to=string(obj(item.get("context")).get("id")),
        sent_at=unix_time(item.get("timestamp")),
    )
