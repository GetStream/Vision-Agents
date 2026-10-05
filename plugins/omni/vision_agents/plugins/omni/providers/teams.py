"""Teams: Bot Framework activities in, activities back out."""

from collections.abc import Mapping

from .._payload import integer, iso_time, obj, objs, string, with_links
from ..message import AttachmentKind, Channel, OmniAttachment, OmniMessage
from ..provider import OmniProvider, Provider

# Everything Teams sends that is not somebody writing: joins, reactions, typing and the
# rest arrive on the same webhook as a message does.
_MESSAGE_TYPE = "message"
# Teams puts a mention of the app in the text as an <at> tag, and the mention itself in
# the activity's entities.
_MENTION = "mention"


class TeamsProvider(OmniProvider):
    """Microsoft Teams, through the Bot Framework."""

    name = Provider.TEAMS
    channels = frozenset({Channel.TEAMS})

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The message in a Bot Framework activity.

        Only `message` activities are read: `conversationUpdate`, `messageReaction`,
        `typing` and `invoke` carry nothing a person wrote. The app's own mention is
        taken out of the text, so an agent reads the question rather than its own name.
        """
        if string(payload.get("type")) != _MESSAGE_TYPE:
            return []
        sender = obj(payload.get("from"))
        recipient = obj(payload.get("recipient"))
        conversation = obj(payload.get("conversation"))
        text = _without_mentions(
            string(payload.get("text")), string(recipient.get("id")), payload
        )
        return [
            OmniMessage(
                channel=Channel.TEAMS,
                provider=self.name,
                conversation_id=string(conversation.get("id")),
                text=text,
                attachments=[
                    _attachment(attachment)
                    for attachment in objs(payload.get("attachments"))
                    # The mention markup rides along as an attachment of its own.
                    if string(attachment.get("contentType")) != "text/html"
                ],
                id=string(payload.get("id")),
                sender_id=string(sender.get("aadObjectId")) or string(sender.get("id")),
                sender_name=string(sender.get("name")),
                account_id=string(recipient.get("id")),
                sent_at=iso_time(payload.get("timestamp")),
            )
        ]

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The activity bodies that send a message to its Teams conversation.

        Files and places go as attachments and links, as they do on Slack: an image with
        a URL is shown, and anything else is a link in the text.
        """
        images = [
            attachment
            for attachment in message.attachments
            if attachment.kind is AttachmentKind.IMAGE and attachment.url
        ]
        links = [
            f"[{attachment.name}]({attachment.link})"
            if attachment.name
            else attachment.link
            for attachment in message.attachments
            if attachment.kind is not AttachmentKind.IMAGE and attachment.link
        ]
        text = with_links(message.text, links)
        if not text and not images:
            return []
        body: dict[str, object] = {"type": _MESSAGE_TYPE, "text": text}
        if text:
            body["textFormat"] = "markdown"
        if images:
            body["attachments"] = [
                {
                    "contentType": image.mime_type or "image/png",
                    "contentUrl": image.url,
                    "name": image.name,
                }
                for image in images
            ]
        return [body]


def _without_mentions(text: str, app_id: str, payload: Mapping[str, object]) -> str:
    """The text with the app's own `<at>` tags taken out."""
    for entity in objs(payload.get("entities")):
        if string(entity.get("type")) != _MENTION:
            continue
        if app_id and string(obj(entity.get("mentioned")).get("id")) != app_id:
            continue
        tag = string(entity.get("text"))
        if tag:
            text = text.replace(tag, "")
    return text.strip()


def _attachment(attachment: dict[str, object]) -> OmniAttachment:
    mime_type = string(attachment.get("contentType"))
    return OmniAttachment(
        kind=AttachmentKind.of(mime_type),
        url=string(attachment.get("contentUrl")),
        mime_type=mime_type,
        name=string(attachment.get("name")),
        size=integer(obj(attachment.get("content")).get("fileSize")),
    )
