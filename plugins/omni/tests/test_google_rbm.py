import base64
import json
from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    GoogleRBMProvider,
)


@pytest.fixture
def user_file() -> dict[str, object]:
    return {
        "senderPhoneNumber": "+12223334444",
        "messageId": "MxABC",
        "sendTime": "2026-10-03T15:01:23.045123456Z",
        "agentId": "plumber_agent@rbm.goog",
        "userFile": {
            "payload": {
                "mimeType": "image/jpeg",
                "fileSizeBytes": 4096,
                "fileUri": "https://rcs-user-content-us.storage.googleapis.com/abc",
                "fileName": "leak.jpg",
            }
        },
    }


@pytest.fixture
def provider() -> GoogleRBMProvider:
    return GoogleRBMProvider()


class TestGoogleRBMProvider:
    def test_parse_reads_a_pubsub_push(
        self, provider: GoogleRBMProvider, user_file: dict[str, object]
    ):
        push = {
            "message": {
                "data": base64.b64encode(json.dumps(user_file).encode()).decode(),
                "messageId": "123",
            },
            "subscription": "projects/p/subscriptions/rbm",
        }

        assert provider.parse(push) == [
            OmniMessage(
                channel=Channel.RCS,
                provider=Provider.GOOGLE_RBM,
                conversation_id="+12223334444",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url="https://rcs-user-content-us.storage.googleapis.com/abc",
                        mime_type="image/jpeg",
                        name="leak.jpg",
                        size=4096,
                    )
                ],
                id="MxABC",
                sender_id="+12223334444",
                account_id="plumber_agent@rbm.goog",
                sent_at=datetime(2026, 10, 3, 15, 1, 23, 45123, tzinfo=timezone.utc),
            )
        ]

    def test_parse_reads_a_suggestion_reply(self, provider: GoogleRBMProvider):
        [message] = provider.parse(
            {
                "senderPhoneNumber": "+12223334444",
                "messageId": "MxDEF",
                "suggestionResponse": {"postbackData": "yes", "text": "Yes please"},
            }
        )

        assert message.text == "Yes please"

    def test_parse_skips_events(self, provider: GoogleRBMProvider):
        event = {
            "senderPhoneNumber": "+12223334444",
            "eventType": "READ",
            "eventId": "E1",
            "messageId": "MxABC",
        }

        assert provider.parse(event) == []

    def test_parse_rejects_data_that_is_not_base64(self, provider: GoogleRBMProvider):
        with pytest.raises(ValueError):
            provider.parse({"message": {"data": "not base64!"}})

    def test_render_text_then_each_file(self, provider: GoogleRBMProvider):
        message = OmniMessage(
            channel=Channel.RCS,
            provider=Provider.GOOGLE_RBM,
            conversation_id="+12223334444",
            text="Booked",
            attachments=[
                OmniAttachment(kind=AttachmentKind.IMAGE, url="https://cdn/a.png"),
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                ),
            ],
        )

        assert provider.render(message) == [
            {
                "contentMessage": {
                    "text": "Booked\nhttps://maps.google.com/?q=52.37,4.89"
                }
            },
            {"contentMessage": {"contentInfo": {"fileUrl": "https://cdn/a.png"}}},
        ]
