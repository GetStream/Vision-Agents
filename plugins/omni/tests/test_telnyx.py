from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    TelnyxProvider,
)


@pytest.fixture
def mms_webhook() -> dict[str, object]:
    return {
        "data": {
            "event_type": "message.received",
            "id": "b301ed3f-1490-491f-995f-6e64e69674d4",
            "occurred_at": "2026-10-04T16:20:00.000+00:00",
            "payload": {
                "id": "84cca175-9755-4859-b67f-4730d7f58aa3",
                "direction": "inbound",
                "type": "MMS",
                "from": {
                    "phone_number": "+15559876543",
                    "carrier": "T-Mobile USA",
                    "line_type": "Wireless",
                },
                "to": [{"phone_number": "+15550001111", "status": "webhook_delivered"}],
                "text": "See photo",
                "media": [
                    {
                        "url": "https://media.telnyx.com/abc.jpg",
                        "content_type": "image/jpeg",
                        "size": 51200,
                    }
                ],
                "received_at": "2026-10-04T16:20:00.000+00:00",
                "messaging_profile_id": "4001",
                "record_type": "message",
            },
            "record_type": "event",
        },
        "meta": {"attempt": 1, "delivered_to": "https://example.com/sms"},
    }


@pytest.fixture
def provider() -> TelnyxProvider:
    return TelnyxProvider()


class TestTelnyxProvider:
    def test_parse_reads_an_mms(
        self, provider: TelnyxProvider, mms_webhook: dict[str, object]
    ):
        assert provider.parse(mms_webhook) == [
            OmniMessage(
                channel=Channel.SMS,
                provider=Provider.TELNYX,
                conversation_id="+15559876543",
                text="See photo",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url="https://media.telnyx.com/abc.jpg",
                        mime_type="image/jpeg",
                        size=51200,
                    )
                ],
                id="84cca175-9755-4859-b67f-4730d7f58aa3",
                sender_id="+15559876543",
                account_id="+15550001111",
                sent_at=datetime(2026, 10, 4, 16, 20, tzinfo=timezone.utc),
            )
        ]

    @pytest.mark.parametrize("event_type", ["message.sent", "message.finalized"])
    def test_parse_skips_delivery_reports(
        self,
        provider: TelnyxProvider,
        mms_webhook: dict[str, object],
        event_type: str,
    ):
        data = mms_webhook["data"]
        assert isinstance(data, dict)
        data["event_type"] = event_type

        assert provider.parse(mms_webhook) == []

    def test_render_batches_media(self, provider: TelnyxProvider):
        message = OmniMessage(
            channel=Channel.SMS,
            provider=Provider.TELNYX,
            conversation_id="+15559876543",
            account_id="+15550001111",
            text="Photos",
            attachments=[
                OmniAttachment(kind=AttachmentKind.IMAGE, url=f"https://cdn/{index}")
                for index in range(11)
            ],
        )

        assert provider.render(message) == [
            {
                "to": "+15559876543",
                "from": "+15550001111",
                "text": "Photos",
                "media_urls": [f"https://cdn/{index}" for index in range(10)],
            },
            {
                "to": "+15559876543",
                "from": "+15550001111",
                "media_urls": ["https://cdn/10"],
            },
        ]

    def test_render_text_with_a_place(self, provider: TelnyxProvider):
        message = OmniMessage(
            channel=Channel.SMS,
            provider=Provider.TELNYX,
            conversation_id="+15559876543",
            text="Meet here",
            attachments=[
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                )
            ],
        )

        assert provider.render(message) == [
            {
                "to": "+15559876543",
                "text": "Meet here\nhttps://maps.google.com/?q=52.37,4.89",
            }
        ]
