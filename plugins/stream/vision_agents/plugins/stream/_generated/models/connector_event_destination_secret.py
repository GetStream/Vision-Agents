from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.connector_event_destination import ConnectorEventDestination


T = TypeVar("T", bound="ConnectorEventDestinationSecret")


@_attrs_define
class ConnectorEventDestinationSecret:
    """An event destination and the secret its forwards are signed with, which no other response carries.

    Attributes:
        destination (ConnectorEventDestination): A URL of the app's own that a connector's raw provider events are
            forwarded to, such as Slack's block_actions or reaction_added. Each forward is a POST of the provider's body as
            it came, with the provider's own Content-Type, signature and timestamp headers, signed on top in the Standard
            Webhooks shape (webhook-id, webhook-timestamp, webhook-signature) with the destination's own secret. A 2xx
            answer is taken; a 5xx, a 429 or no answer is sent again after 5 s, 5 min, 30 min and 2 h; any other answer is
            not sent again. The provider's signature headers come only while the provider's own check would pass them: for
            Slack, until X-Slack-Request-Timestamp is 5 minutes old, the age Slack Bolt refuses after. A forward sent later,
            such as the retries after 5 min, 30 min and 2 h, carries Content-Type alone of them: verify it with webhook-
            signature. webhook-id is the same for every delivery of one provider event (Slack's event_id, or trigger_id for
            an interaction), and a digest of the body for one that names no id.
        secret (str): The Standard Webhooks signing secret, whsec_ and 32 random bytes in base64. Shown this once: keep
            it, no later response carries it.
    """

    destination: ConnectorEventDestination
    secret: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        destination = self.destination.to_dict()

        secret = self.secret

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "destination": destination,
                "secret": secret,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_event_destination import ConnectorEventDestination

        d = dict(src_dict)
        destination = ConnectorEventDestination.from_dict(d.pop("destination"))

        secret = d.pop("secret")

        connector_event_destination_secret = cls(
            destination=destination,
            secret=secret,
        )

        connector_event_destination_secret.additional_properties = d
        return connector_event_destination_secret

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
