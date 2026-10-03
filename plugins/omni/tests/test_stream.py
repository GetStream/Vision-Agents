from datetime import datetime, timezone

import pytest
from getstream.models import Attachment, MessageRequest

from vision_agents.plugins.omni import (
    OMNI_KEY,
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    from_stream,
    stream_channel_id,
    stream_user_id,
    to_stream,
)


@pytest.fixture
def inbound() -> OmniMessage:
    return OmniMessage(
        channel=Channel.SMS,
        conversation_id="+15559876543",
        text="Here is the leak",
        attachments=[
            OmniAttachment(
                kind=AttachmentKind.IMAGE,
                url="https://api.twilio.com/media/ME1",
                mime_type="image/jpeg",
                size=2048,
            ),
            OmniAttachment(
                kind=AttachmentKind.FILE,
                url="https://example.com/quote.pdf",
                mime_type="application/pdf",
                name="quote.pdf",
            ),
            OmniAttachment(
                kind=AttachmentKind.LOCATION,
                name="Home",
                latitude=52.37,
                longitude=0.0,
            ),
        ],
        id="SM123",
        sender_id="+15559876543",
        account_id="+15550001111",
        sent_at=datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc),
    )


class TestStream:
    def test_to_stream_maps_text_attachments_and_custom(self, inbound: OmniMessage):
        request = to_stream(inbound)

        assert request.text == "Here is the leak"
        assert request.user_id == "sms__15559876543"
        assert request.attachments == [
            Attachment(
                type="image",
                image_url="https://api.twilio.com/media/ME1",
                custom={"mime_type": "image/jpeg", "file_size": 2048},
            ),
            Attachment(
                type="file",
                asset_url="https://example.com/quote.pdf",
                title="quote.pdf",
                custom={"mime_type": "application/pdf"},
            ),
            Attachment(
                type="location",
                title="Home",
                title_link="https://maps.google.com/?q=52.37,0.0",
                custom={"latitude": 52.37, "longitude": 0.0},
            ),
        ]
        assert request.custom == {
            OMNI_KEY: {
                "v": 1,
                "channel": "sms",
                "conversation_id": "+15559876543",
                "id": "SM123",
                "sender_id": "+15559876543",
                "account_id": "+15550001111",
                "sent_at": "2026-10-03T12:00:00+00:00",
            }
        }

    def test_to_stream_uses_given_user(self, inbound: OmniMessage):
        assert to_stream(inbound, user_id="customer-1").user_id == "customer-1"

    def test_round_trip_through_json(self, inbound: OmniMessage):
        request = MessageRequest.from_dict(to_stream(inbound).to_dict())

        assert from_stream(request) == inbound

    def test_from_stream_with_route_answers_the_conversation(
        self, inbound: OmniMessage
    ):
        inbound.thread_id = "1700000000.000100"
        reply = MessageRequest(
            text="A plumber is on the way",
            attachments=[
                Attachment(type="ai_reasoning", custom={"summary": "Thinking"}),
                Attachment(type="image", image_url="https://cdn/map.png", custom={}),
            ],
        )

        assert from_stream(reply, route=inbound) == OmniMessage(
            channel=Channel.SMS,
            conversation_id="+15559876543",
            account_id="+15550001111",
            thread_id="1700000000.000100",
            text="A plumber is on the way",
            attachments=[
                OmniAttachment(kind=AttachmentKind.IMAGE, url="https://cdn/map.png")
            ],
        )

    def test_from_stream_without_route_or_custom_fails(self):
        with pytest.raises(ValueError, match="custom.omni"):
            from_stream(MessageRequest(text="hello"))

    def test_ids_are_valid_stream_ids(self):
        message = OmniMessage(
            channel=Channel.IMESSAGE,
            conversation_id="8f392755-6865-4b18-880a-227f9d8b458f",
            sender_id="person@icloud.com",
        )

        assert stream_user_id(message) == "imessage_person_icloud_com"
        assert (
            stream_channel_id(message)
            == "imessage_8f392755-6865-4b18-880a-227f9d8b458f"
        )
