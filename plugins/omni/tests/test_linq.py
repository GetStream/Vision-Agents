from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    linq,
)


@pytest.fixture
def received() -> dict[str, object]:
    return {
        "api_version": "v3",
        "webhook_version": "2026-02-03",
        "event_type": "message.received",
        "event_id": "2915e81c-5068-4796-ace2-21d2c94ad298",
        "created_at": "2026-02-05T19:31:13.736444093Z",
        "partner_id": "your-partner-id",
        "data": {
            "chat": {
                "id": "8f392755-6865-4b18-880a-227f9d8b458f",
                "is_group": False,
                "owner_handle": {
                    "handle": "+12025551234",
                    "id": "6d6c617f-187a-4dcd-a0d5-988347a8c092",
                    "is_me": True,
                    "service": "iMessage",
                },
            },
            "id": "89e3566e-1d13-49e5-a8ee-48490d5bfeb7",
            "direction": "inbound",
            "sender_handle": {
                "handle": "+12025559876",
                "id": "e604375a-5913-483a-8278-c631e8f0ffda",
                "is_me": False,
                "service": "iMessage",
            },
            "parts": [
                {"type": "text", "value": "Hello!"},
                {
                    "type": "media",
                    "id": "f13dda7d-ecac-49eb-b3fe-16fe286abf19",
                    "filename": "photo.jpg",
                    "mime_type": "image/jpeg",
                    "size_bytes": 245678,
                    "url": "https://cdn.linqapp.com/attachments/a1b2c3d4/photo.jpg",
                },
            ],
            "effect": None,
            "reply_to": {"message_id": "347d62c2-2170-4754-8d30-c76d0c727d96"},
            "sent_at": "2026-02-05T19:31:13.074Z",
            "service": "iMessage",
        },
    }


class TestLinq:
    def test_parse_reads_a_received_message(self, received: dict[str, object]):
        assert linq.parse(received) == [
            OmniMessage(
                channel=Channel.IMESSAGE,
                conversation_id="8f392755-6865-4b18-880a-227f9d8b458f",
                text="Hello!",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url="https://cdn.linqapp.com/attachments/a1b2c3d4/photo.jpg",
                        media_id="f13dda7d-ecac-49eb-b3fe-16fe286abf19",
                        mime_type="image/jpeg",
                        name="photo.jpg",
                        size=245678,
                    )
                ],
                id="89e3566e-1d13-49e5-a8ee-48490d5bfeb7",
                sender_id="+12025559876",
                account_id="+12025551234",
                reply_to="347d62c2-2170-4754-8d30-c76d0c727d96",
                sent_at=datetime(2026, 2, 5, 19, 31, 13, 74000, tzinfo=timezone.utc),
            )
        ]

    def test_parse_skips_other_events(self, received: dict[str, object]):
        received["event_type"] = "message.delivered"

        assert linq.parse(received) == []

    def test_render_text_part_then_media(self):
        message = OmniMessage(
            channel=Channel.IMESSAGE,
            conversation_id="8f392755-6865-4b18-880a-227f9d8b458f",
            text="Here you go",
            reply_to="89e3566e-1d13-49e5-a8ee-48490d5bfeb7",
            attachments=[
                OmniAttachment(kind=AttachmentKind.FILE, url="https://cdn/q.pdf"),
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                ),
            ],
        )

        assert linq.render(message) == [
            {
                "message": {
                    "parts": [
                        {
                            "type": "text",
                            "value": "Here you go\nhttps://maps.google.com/?q=52.37,4.89",
                        },
                        {"type": "media", "url": "https://cdn/q.pdf"},
                    ],
                    "reply_to": {"message_id": "89e3566e-1d13-49e5-a8ee-48490d5bfeb7"},
                }
            }
        ]
