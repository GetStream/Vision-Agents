from datetime import datetime, timezone

import pytest

from vision_agents.plugins.omni import (
    AttachmentKind,
    Channel,
    OmniAttachment,
    OmniMessage,
    Provider,
    TeamsProvider,
)


@pytest.fixture
def mention_activity() -> dict[str, object]:
    return {
        "type": "message",
        "id": "1759593600000",
        "timestamp": "2026-10-04T16:20:00.0000000Z",
        "serviceUrl": "https://smba.trafficmanager.net/emea/",
        "channelId": "msteams",
        "from": {
            "id": "29:1abc",
            "name": "Ada Lovelace",
            "aadObjectId": "8f1e0c2a-5d44-4b19-9a1e-7f0b9c2d3e4f",
        },
        "conversation": {"conversationType": "channel", "id": "19:team@thread.tacv2"},
        "recipient": {"id": "28:bot", "name": "On call"},
        "textFormat": "plain",
        "text": "<at>On call</at> what broke?",
        "attachments": [
            {
                "contentType": "text/html",
                "content": "<div><span itemscope>On call</span> what broke?</div>",
            },
            {
                "contentType": "image/png",
                "contentUrl": "https://teams.microsoft.com/files/graph.png",
                "name": "graph.png",
                "content": {"fileSize": 20480},
            },
        ],
        "entities": [
            {
                "type": "mention",
                "text": "<at>On call</at>",
                "mentioned": {"id": "28:bot", "name": "On call"},
            }
        ],
    }


@pytest.fixture
def provider() -> TeamsProvider:
    return TeamsProvider()


class TestTeamsProvider:
    def test_parse_reads_a_mention_without_the_apps_own_name(
        self, provider: TeamsProvider, mention_activity: dict[str, object]
    ):
        assert provider.parse(mention_activity) == [
            OmniMessage(
                channel=Channel.TEAMS,
                provider=Provider.TEAMS,
                conversation_id="19:team@thread.tacv2",
                text="what broke?",
                attachments=[
                    OmniAttachment(
                        kind=AttachmentKind.IMAGE,
                        url="https://teams.microsoft.com/files/graph.png",
                        mime_type="image/png",
                        name="graph.png",
                        size=20480,
                    )
                ],
                id="1759593600000",
                sender_id="8f1e0c2a-5d44-4b19-9a1e-7f0b9c2d3e4f",
                sender_name="Ada Lovelace",
                account_id="28:bot",
                sent_at=datetime(2026, 10, 4, 16, 20, tzinfo=timezone.utc),
            )
        ]

    def test_parse_keeps_a_mention_of_somebody_else(
        self, provider: TeamsProvider, mention_activity: dict[str, object]
    ):
        entities = mention_activity["entities"]
        assert isinstance(entities, list)
        entities[0]["mentioned"] = {"id": "29:2def", "name": "Grace"}
        mention_activity["text"] = "<at>Grace</at> what broke?"

        assert provider.parse(mention_activity)[0].text == "<at>Grace</at> what broke?"

    @pytest.mark.parametrize(
        "activity_type", ["conversationUpdate", "messageReaction", "typing", "invoke"]
    )
    def test_parse_skips_what_nobody_wrote(
        self,
        provider: TeamsProvider,
        mention_activity: dict[str, object],
        activity_type: str,
    ):
        mention_activity["type"] = activity_type

        assert provider.parse(mention_activity) == []

    def test_render_sends_markdown_with_an_image(self, provider: TeamsProvider):
        message = OmniMessage(
            channel=Channel.TEAMS,
            provider=Provider.TEAMS,
            conversation_id="19:team@thread.tacv2",
            text="Here it is",
            attachments=[
                OmniAttachment(
                    kind=AttachmentKind.IMAGE,
                    url="https://cdn/render.png",
                    mime_type="image/png",
                    name="render.png",
                )
            ],
        )

        assert provider.render(message) == [
            {
                "type": "message",
                "text": "Here it is",
                "textFormat": "markdown",
                "attachments": [
                    {
                        "contentType": "image/png",
                        "contentUrl": "https://cdn/render.png",
                        "name": "render.png",
                    }
                ],
            }
        ]

    def test_render_links_a_file_and_a_place(self, provider: TeamsProvider):
        message = OmniMessage(
            channel=Channel.TEAMS,
            provider=Provider.TEAMS,
            conversation_id="19:team@thread.tacv2",
            text="Both of these",
            attachments=[
                OmniAttachment(
                    kind=AttachmentKind.FILE,
                    url="https://cdn/report.pdf",
                    name="report.pdf",
                ),
                OmniAttachment(
                    kind=AttachmentKind.LOCATION, latitude=52.37, longitude=4.89
                ),
            ],
        )

        assert provider.render(message) == [
            {
                "type": "message",
                "text": "Both of these\n[report.pdf](https://cdn/report.pdf)\n"
                "https://maps.google.com/?q=52.37,4.89",
                "textFormat": "markdown",
            }
        ]

    def test_render_says_nothing_about_an_empty_message(self, provider: TeamsProvider):
        message = OmniMessage(
            channel=Channel.TEAMS,
            provider=Provider.TEAMS,
            conversation_id="19:team@thread.tacv2",
        )

        assert provider.render(message) == []
