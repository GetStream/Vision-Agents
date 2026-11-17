import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    TwilioProvider,
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


@pytest.fixture
def provider() -> TwilioProvider:
    return TwilioProvider()


class TestTwilioProvider:
    def test_parse_reads_an_mms(
        self, provider: TwilioProvider, mms_form: dict[str, str]
    ):
        assert provider.parse(mms_form) == [
            OmniMessage(
                channel=Channel.SMS,
                provider=Provider.TWILIO,
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

    @pytest.mark.parametrize(
        ("sender", "channel"),
        [
            ("+15559876543", Channel.SMS),
            ("whatsapp:+15559876543", Channel.WHATSAPP),
            ("rcs:+15559876543", Channel.RCS),
        ],
    )
    def test_parse_reads_the_channel_from_the_sender(
        self,
        provider: TwilioProvider,
        mms_form: dict[str, str],
        sender: str,
        channel: Channel,
    ):
        mms_form["From"] = sender

        [message] = provider.parse(mms_form)

        assert message.channel is channel
        assert message.conversation_id == sender

    def test_parse_skips_status_callbacks(self, provider: TwilioProvider):
        callback = {"MessageSid": "SM1", "MessageStatus": "delivered", "To": "+1555"}

        assert provider.parse(callback) == []

    def test_render_batches_media(self, provider: TwilioProvider):
        message = OmniMessage(
            channel=Channel.SMS,
            provider=Provider.TWILIO,
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

    def test_render_text_with_a_place(self, provider: TwilioProvider):
        message = OmniMessage(
            channel=Channel.SMS,
            provider=Provider.TWILIO,
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
                "To": "+15559876543",
                "Body": "Meet here\nhttps://maps.google.com/?q=52.37,4.89",
            }
        ]
