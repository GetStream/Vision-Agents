"""Finding the provider that reads a webhook or sends a message."""

from collections.abc import Mapping
from typing import Optional

from .message import OmniMessage
from .provider import OmniProvider
from .providers import (
    GoogleRBMProvider,
    LinqProvider,
    SlackProvider,
    TeamsProvider,
    TelnyxProvider,
    TwilioProvider,
    WhatsAppProvider,
)


class UnknownProviderError(KeyError):
    """No provider is registered under a name."""


class ProviderRegistry:
    """The providers an app talks to, by name.

    Providers keep state, such as the Slack messages already seen, so keep one registry
    for the life of the app.
    """

    def __init__(self, providers: Optional[list[OmniProvider]] = None):
        """Registers the given providers, or one of each built-in provider.

        Args:
            providers: The providers to start with.
        """
        self._providers: dict[str, OmniProvider] = {}
        for provider in (
            providers
            if providers is not None
            else [
                SlackProvider(),
                TeamsProvider(),
                WhatsAppProvider(),
                GoogleRBMProvider(),
                TwilioProvider(),
                TelnyxProvider(),
                LinqProvider(),
            ]
        ):
            self.register(provider)

    def register(self, provider: OmniProvider) -> None:
        """Adds a provider, replacing any registered under its name."""
        self._providers[provider.name] = provider

    def get(self, name: str) -> OmniProvider:
        """The provider registered under a name.

        Raises:
            UnknownProviderError: None is.
        """
        try:
            return self._providers[name]
        except KeyError:
            raise UnknownProviderError(name) from None

    def parse(self, name: str, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The messages in a webhook body sent by the provider registered under a name."""
        return self.get(name).parse(payload)

    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The bodies that send a message through its provider.

        Raises:
            UnknownProviderError: Its provider is not registered.
            ValueError: Its provider does not carry its channel.
        """
        provider = self.get(message.provider)
        if message.channel not in provider.channels:
            raise ValueError(
                f"provider {message.provider} does not carry {message.channel.value}"
            )
        return provider.render(message)
