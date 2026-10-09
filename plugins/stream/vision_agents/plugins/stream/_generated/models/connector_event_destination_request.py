from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connector_event_forward import ConnectorEventForward

T = TypeVar("T", bound="ConnectorEventDestinationRequest")


@_attrs_define
class ConnectorEventDestinationRequest:
    """An event destination to create. An unknown field is refused rather than ignored.

    Attributes:
        forward (ConnectorEventForward): Which deliveries a destination is sent. unhandled: the ones the router acts on
            in no way, such as a Slack button click, a reaction or a modal submission, and a message no agent of the app
            answers: the app's own code next to the router's agent. all: every verified delivery, messages and grant events
            included, but the provider's URL handshake: the app runs its own agent. Either way a message an agent of the app
            answers is still answered there. A Slack reply in a thread the agent is not in yet is unhandled when it arrives.
            If the mention that starts the thread arrives after it, as when Slack retries the mention, the agent answers the
            reply too. The forward is not taken back, and no event says that the agent answered it.
        url (str): A public https URL. One that is or resolves to a private, loopback or link-local address is refused.
    """

    forward: ConnectorEventForward
    url: str

    def to_dict(self) -> dict[str, Any]:
        forward = self.forward.value

        url = self.url

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "forward": forward,
                "url": url,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        forward = ConnectorEventForward(d.pop("forward"))

        url = d.pop("url")

        connector_event_destination_request = cls(
            forward=forward,
            url=url,
        )

        return connector_event_destination_request
