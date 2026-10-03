import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    sms,
)


@pytest.fixture
def mms_form() -> dict[str, str]:
    return {
        "ToCountry": "US",
        "SmsMessageSid": "MM123",
        "NumMedia": "2",
        "SmsSid": "MM123",
        "SmsStatus": "received",
        "Body": "See photos",
        "To": "+15550001111",
        "MessageSid": "MM123",
        "AccountSid": "AC123",
        "From": "+15559876543",
        "MediaContentType0": "image/jpeg",
        "MediaUrl0": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM123/Media/ME0",
        "MediaContentType1": "video/mp4",
        "MediaUrl1": "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages/MM123/Media/ME1",
        "ApiVersion": "2010-04-01",
    }


class TestSms:
    def test_parse_reads_an_mms(self, mms_form: dict[str, str]):
        assert sms.parse(mms_form) == [
            OmniMessage(
                channel=Channel.SMS,
                conversation_id="+15559876543",
                text="See photos",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url=mms_form["MediaUrl0"],
                        mime_type="image/jpeg",
                    ),
                    OmniAttachment(
                        kind=AttachmentKind.VIDEO,
                        url=mms_form["MediaUrl1"],
                        mime_type="video/mp4",
                    ),
                ],
                id="MM123",
                sender_id="+15559876543",
                account_id="+15550001111",
            )
        ]

    def test_parse_skips_status_callbacks(self):
        callback = {"MessageSid": "SM1", "MessageStatus": "delivered", "To": "+1555"}

        assert sms.parse(callback) == []

    def test_render_batches_media(self):
        message = OmniMessage(
            channel=Channel.SMS,
            conversation_id="+15559876543",
            account_id="+15550001111",
            text="Photos",
            attachments=[
                OmniAttachment(kind=AttachmentKind.IMAGE, url=f"https://cdn/{index}")
                for index in range(11)
            ],
        )

        assert sms.render(message) == [
            {
                "To": "+15559876543",
                "From": "+15550001111",
                "Body": "Photos",
                "MediaUrl": [f"https://cdn/{index}" for index in range(10)],
            },
            {
                "To": "+15559876543",
                "From": "+15550001111",
                "MediaUrl": ["https://cdn/10"],
            },
        ]

    def test_render_text_with_a_place(self):
        message = OmniMessage(
            channel=Channel.SMS,
            conversation_id="+15559876543",
            text="Meet here",
            attachments=[
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                )
            ],
        )

        assert sms.render(message) == [
            {
                "To": "+15559876543",
                "Body": "Meet here\nhttps://maps.google.com/?q=52.37,4.89",
            }
        ]
