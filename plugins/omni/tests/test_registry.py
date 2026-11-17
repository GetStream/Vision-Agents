from collections.abc import Mapping

import pytest

from vision_agents.plugins.omni import (
    Channel,
    OmniMessage,
    OmniProvider,
    Provider,
    ProviderRegistry,
    UnknownProviderError,
)


class EchoProvider(OmniProvider):
    """A provider of SMS that sends the text back as it is."""

    name = "echo"
    channels = frozenset({Channel.SMS})

    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        return [
            OmniMessage(
                channel=Channel.SMS,
                provider=self.name,
                conversation_id=str(payload["from"]),
                text=str(payload["text"]),
            )
        ]

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        return [{"to": message.conversation_id, "text": message.text}]


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


class TestProviderRegistry:
    def test_has_every_built_in_provider(self, registry: ProviderRegistry):
        carried = {
            name: registry.get(name).channels
            for name in (
                Provider.SLACK,
                Provider.TEAMS,
                Provider.WHATSAPP,
                Provider.GOOGLE_RBM,
                Provider.TWILIO,
                Provider.TELNYX,
                Provider.LINQ,
            )
        }

        assert carried == {
            "slack": {Channel.SLACK},
            "teams": {Channel.TEAMS},
            "whatsapp": {Channel.WHATSAPP},
            "google_rbm": {Channel.RCS},
            "twilio": {Channel.SMS, Channel.WHATSAPP, Channel.RCS},
            "telnyx": {Channel.SMS},
            "linq": {Channel.IMESSAGE, Channel.RCS, Channel.SMS},
        }

    def test_parses_and_renders_through_the_message_provider(
        self, registry: ProviderRegistry
    ):
        [message] = registry.parse(
            Provider.TWILIO,
            {
                "MessageSid": "SM1",
                "From": "+15559876543",
                "To": "+15550001111",
                "Body": "Hi",
            },
        )

        assert registry.render(message) == [
            {"To": "+15559876543", "From": "+15550001111", "Body": "Hi"}
        ]

    def test_registers_a_provider_of_your_own(self, registry: ProviderRegistry):
        registry.register(EchoProvider())

        [message] = registry.parse("echo", {"from": "+15559876543", "text": "Hi"})

        assert registry.render(message) == [{"to": "+15559876543", "text": "Hi"}]

    def test_starts_with_only_the_given_providers(self):
        registry = ProviderRegistry([EchoProvider()])

        with pytest.raises(UnknownProviderError):
            registry.get(Provider.SLACK)

    def test_rejects_an_unknown_provider(self, registry: ProviderRegistry):
        message = OmniMessage(
            channel=Channel.SMS, provider="carrier-pigeon", conversation_id="+1555"
        )

        with pytest.raises(UnknownProviderError):
            registry.render(message)

    def test_rejects_a_channel_the_provider_does_not_carry(
        self, registry: ProviderRegistry
    ):
        message = OmniMessage(
            channel=Channel.IMESSAGE,
            provider=Provider.TWILIO,
            conversation_id="+15559876543",
            text="Hi",
        )

        with pytest.raises(ValueError, match="does not carry imessage"):
            registry.render(message)
