import copy
from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    SlackProvider,
)


@pytest.fixture
def event_callback() -> dict[str, object]:
    return {
        "token": "XXYYZZ",
        "team_id": "T0001",
        "api_app_id": "A0001",
        "type": "event_callback",
        "event_id": "Ev0001",
        "event_time": 1700000000,
        "event": {
            "type": "message",
            "subtype": "file_share",
            "channel": "C0001",
            "channel_type": "channel",
            "user": "U0001",
            "text": "Is this broken?",
            "ts": "1700000000.000200",
            "thread_ts": "1700000000.000100",
            "files": [
                {
                    "id": "F0001",
                    "name": "sink.png",
                    "mimetype": "image/png",
                    "size": 1234,
                    "url_private": "https://files.slack.com/files-pri/T0001-F0001/sink.png",
                }
            ],
        },
    }


@pytest.fixture
def provider() -> SlackProvider:
    return SlackProvider()


class TestSlackProvider:
    def test_parse_reads_a_message(
        self, provider: SlackProvider, event_callback: dict[str, object]
    ):
        assert provider.parse(event_callback) == [
            OmniMessage(
                channel=Channel.SLACK,
                provider=Provider.SLACK,
                conversation_id="C0001",
                text="Is this broken?",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url="https://files.slack.com/files-pri/T0001-F0001/sink.png",
                        media_id="F0001",
                        mime_type="image/png",
                        name="sink.png",
                        size=1234,
                    )
                ],
                id="1700000000.000200",
                sender_id="U0001",
                account_id="T0001",
                thread_id="1700000000.000100",
                sent_at=datetime(2023, 11, 14, 22, 13, 20, 200, tzinfo=timezone.utc),
            )
        ]

    @pytest.mark.parametrize(
        "change",
        [
            {"bot_id": "B0001"},
            {"subtype": "message_changed"},
            {"type": "reaction_added"},
        ],
    )
    def test_parse_skips_what_a_person_did_not_write(
        self,
        provider: SlackProvider,
        event_callback: dict[str, object],
        change: dict[str, object],
    ):
        event = event_callback["event"]
        assert isinstance(event, dict)
        event.update(change)

        assert provider.parse(event_callback) == []

    def test_parse_reads_a_message_once(
        self, provider: SlackProvider, event_callback: dict[str, object]
    ):
        mention = copy.deepcopy(event_callback)
        event = mention["event"]
        assert isinstance(event, dict)
        event["type"] = "app_mention"
        del event["subtype"]

        assert len(provider.parse(event_callback)) == 1
        assert provider.parse(mention) == []
        assert provider.parse(event_callback) == []

    def test_parse_forgets_the_oldest_messages(self, event_callback: dict[str, object]):
        provider = SlackProvider(remembered=1)
        later = copy.deepcopy(event_callback)
        event = later["event"]
        assert isinstance(event, dict)
        event["ts"] = "1700000001.000000"

        provider.parse(event_callback)
        provider.parse(later)

        assert len(provider.parse(event_callback)) == 1

    def test_parse_skips_url_verification(self, provider: SlackProvider):
        assert provider.parse({"type": "url_verification", "challenge": "abc"}) == []

    def test_render_text_in_thread(self, provider: SlackProvider):
        message = OmniMessage(
            channel=Channel.SLACK,
            provider=Provider.SLACK,
            conversation_id="C0001",
            text="On it",
            thread_id="1700000000.000100",
        )

        assert provider.render(message) == [
            {"channel": "C0001", "text": "On it", "thread_ts": "1700000000.000100"}
        ]

    def test_render_images_as_blocks_and_files_as_links(self, provider: SlackProvider):
        message = OmniMessage(
            channel=Channel.SLACK,
            provider=Provider.SLACK,
            conversation_id="D0001",
            text="Your quote",
            attachments=[
                OmniAttachment(kind=AttachmentKind.IMAGE, url="https://cdn/a.png"),
                OmniAttachment(
                    kind=AttachmentKind.FILE, url="https://cdn/q.pdf", name="q.pdf"
                ),
                OmniAttachment(kind=AttachmentKind.VIDEO, media_id="only-an-id"),
            ],
        )
        text = "Your quote\n<https://cdn/q.pdf|q.pdf>"

        assert provider.render(message) == [
            {
                "channel": "D0001",
                "text": text,
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                    {
                        "type": "image",
                        "image_url": "https://cdn/a.png",
                        "alt_text": "image",
                    },
                ],
            }
        ]

    def test_render_nothing_to_send(self, provider: SlackProvider):
        assert (
            provider.render(
                OmniMessage(
                    channel=Channel.SLACK, provider=Provider.SLACK, conversation_id="C1"
                )
            )
            == []
        )
