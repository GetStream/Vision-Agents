import dataclasses
import re
from datetime import datetime, timezone

import pytest
from getstream.models import Attachment, MessageRequest

from vision_agents.plugins.omni import (
    OMNI_KEY,
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    from_stream,
    stream_channel_id,
    stream_user_id,
    to_stream,
)


@pytest.fixture
def inbound() -> OmniMessage:
    return OmniMessage(
        channel=Channel.SMS,
        provider=Provider.TWILIO,
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
        assert request.user_id == stream_user_id(inbound)
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
                "provider": "twilio",
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
            provider=Provider.TWILIO,
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

    def test_ids_are_short_valid_stream_ids(self):
        message = OmniMessage(
            channel=Channel.IMESSAGE,
            provider=Provider.LINQ,
            conversation_id="8f392755-6865-4b18-880a-227f9d8b458f",
            account_id="a-very-long-line-handle-" * 4,
            sender_id="person.with+a.long-address@icloud.com" * 4,
        )

        for stream_id in (stream_user_id(message), stream_channel_id(message)):
            assert re.fullmatch(r"linq_[0-9a-f]{24}", stream_id)

    def test_channel_id_splits_by_account_not_by_channel(self, inbound: OmniMessage):
        other_number = dataclasses.replace(inbound, account_id="+15550002222")
        fallen_back = dataclasses.replace(inbound, channel=Channel.RCS)

        assert stream_channel_id(other_number) != stream_channel_id(inbound)
        assert stream_channel_id(fallen_back) == stream_channel_id(inbound)

    def test_user_id_is_one_per_sender(self, inbound: OmniMessage):
        on_other_number = dataclasses.replace(inbound, account_id="+15550002222")
        someone_else = dataclasses.replace(inbound, sender_id="+15551112222")

        assert stream_user_id(on_other_number) == stream_user_id(inbound)
        assert stream_user_id(someone_else) != stream_user_id(inbound)
