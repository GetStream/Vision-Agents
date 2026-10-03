from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    whatsapp,
)


@pytest.fixture
def webhook() -> dict[str, object]:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15550001111",
                                "phone_number_id": "PHONE_NUMBER_ID",
                            },
                            "contacts": [
                                {"profile": {"name": "Kerry"}, "wa_id": "16315551181"}
                            ],
                            "messages": [
                                {
                                    "from": "16315551181",
                                    "id": "wamid.TEXT",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": "Hello"},
                                },
                                {
                                    "from": "16315551181",
                                    "id": "wamid.IMAGE",
                                    "timestamp": "1700000001",
                                    "type": "image",
                                    "context": {
                                        "from": "15550001111",
                                        "id": "wamid.EARLIER",
                                    },
                                    "image": {
                                        "caption": "The leak",
                                        "mime_type": "image/jpeg",
                                        "sha256": "abc",
                                        "id": "MEDIA_ID",
                                    },
                                },
                                {
                                    "from": "16315551181",
                                    "id": "wamid.LOCATION",
                                    "timestamp": "1700000002",
                                    "type": "location",
                                    "location": {
                                        "latitude": 52.37,
                                        "longitude": 4.89,
                                        "name": "Home",
                                    },
                                },
                                {
                                    "from": "16315551181",
                                    "id": "wamid.BUTTON",
                                    "timestamp": "1700000003",
                                    "type": "interactive",
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {"id": "yes", "title": "Yes"},
                                    },
                                },
                                {
                                    "from": "16315551181",
                                    "id": "wamid.REACTION",
                                    "timestamp": "1700000004",
                                    "type": "reaction",
                                    "reaction": {
                                        "message_id": "wamid.X",
                                        "emoji": "👍",
                                    },
                                },
                            ],
                        },
                    }
                ],
            }
        ],
    }


class TestWhatsapp:
    def test_parse_reads_messages(self, webhook: dict[str, object]):
        sender = {
            "channel": Channel.WHATSAPP,
            "conversation_id": "16315551181",
            "sender_id": "16315551181",
            "sender_name": "Kerry",
            "account_id": "PHONE_NUMBER_ID",
        }

        assert whatsapp.parse(webhook) == [
            OmniMessage(
                text="Hello",
                id="wamid.TEXT",
                sent_at=datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc),
                **sender,
            ),
            OmniMessage(
                text="The leak",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        media_id="MEDIA_ID",
                        mime_type="image/jpeg",
                    )
                ],
                id="wamid.IMAGE",
                reply_to="wamid.EARLIER",
                sent_at=datetime(2023, 11, 14, 22, 13, 21, tzinfo=timezone.utc),
                **sender,
            ),
            OmniMessage(
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.LOCATION,
                        name="Home",
                        latitude=52.37,
                        longitude=4.89,
                    )
                ],
                id="wamid.LOCATION",
                sent_at=datetime(2023, 11, 14, 22, 13, 22, tzinfo=timezone.utc),
                **sender,
            ),
            OmniMessage(
                text="Yes",
                id="wamid.BUTTON",
                sent_at=datetime(2023, 11, 14, 22, 13, 23, tzinfo=timezone.utc),
                **sender,
            ),
        ]

    def test_parse_skips_statuses(self):
        webhook = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "statuses": [{"id": "wamid.X", "status": "delivered"}]
                            }
                        }
                    ]
                }
            ]
        }

        assert whatsapp.parse(webhook) == []

    def test_render_one_body_per_part(self):
        message = OmniMessage(
            channel=Channel.WHATSAPP,
            conversation_id="16315551181",
            text="Here is your quote",
            reply_to="wamid.EARLIER",
            attachments=[
                OmniAttachment(
                    kind=AttachmentKind.FILE, url="https://cdn/q.pdf", name="q.pdf"
                ),
                OmniAttachment(kind=AttachmentKind.IMAGE, media_id="MEDIA_ID"),
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                ),
                OmniAttachment(kind=AttachmentKind.AUDIO),
            ],
        )
        envelope = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": "16315551181",
            "context": {"message_id": "wamid.EARLIER"},
        }

        assert whatsapp.render(message) == [
            {"type": "text", "text": {"body": "Here is your quote"}, **envelope},
            {
                "type": "document",
                "document": {"link": "https://cdn/q.pdf", "filename": "q.pdf"},
                **envelope,
            },
            {"type": "image", "image": {"id": "MEDIA_ID"}, **envelope},
            {
                "type": "location",
                "location": {"latitude": 52.37, "longitude": 4.89},
                **envelope,
            },
        ]
