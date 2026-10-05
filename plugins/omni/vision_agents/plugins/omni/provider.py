"""What a messaging provider does for omni: read its webhooks and write its sends."""

import abc
from collections.abc import Mapping

from .message import Channel, OmniMessage


class Provider:
    """The names of the providers omni ships with.

    A provider of your own needs only a unique name of letters, digits, `_` and `-`, as
    it starts the Stream ids of its users and channels.
    """

    SLACK = "slack"
    TEAMS = "teams"
    WHATSAPP = "whatsapp"
    GOOGLE_RBM = "google_rbm"
    TWILIO = "twilio"
    TELNYX = "telnyx"
    LINQ = "linq"


class OmniProvider(abc.ABC):
    """A provider of one or more channels.

    Attributes:
        name: Its name, which messages record as their `provider`.
        channels: The channels it carries.
    """

    name: str
    channels: frozenset[Channel]

    @abc.abstractmethod
    def parse(self, payload: Mapping[str, object]) -> list[OmniMessage]:
        """The messages a person wrote in one of its webhook bodies.

        Args:
            payload: The webhook body, or its form fields.

        Returns:
            The messages, in order. None for webhooks that carry no message.
        """

    @abc.abstractmethod
    def render(self, message: OmniMessage) -> list[dict[str, object]]:
        """The bodies of the API requests that send a message, in the order to send them.

        Args:
            message: The message, addressed by its `conversation_id` and `account_id`.

        Returns:
            The bodies. None for a message with nothing to send.
        """
