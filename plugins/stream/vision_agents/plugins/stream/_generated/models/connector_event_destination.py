from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_event_forward import ConnectorEventForward
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorEventDestination")


@_attrs_define
class ConnectorEventDestination:
    """A URL of the app's own that a connector's raw provider events are forwarded to, such as Slack's block_actions or
    reaction_added. Each forward is a POST of the provider's body as it came, with the provider's own Content-Type,
    signature and timestamp headers, signed on top in the Standard Webhooks shape (webhook-id, webhook-timestamp,
    webhook-signature) with the destination's own secret. A 2xx answer is taken; a 5xx, a 429 or no answer is sent again
    after 5 s, 5 min, 30 min and 2 h; any other answer is not sent again. The provider's signature headers come only
    while the provider's own check would pass them: for Slack, until X-Slack-Request-Timestamp is 5 minutes old, the age
    Slack Bolt refuses after. A forward sent later, such as the retries after 5 min, 30 min and 2 h, carries Content-
    Type alone of them: verify it with webhook-signature. webhook-id is the same for every delivery of one provider
    event (Slack's event_id, or trigger_id for an interaction), and a digest of the body for one that names no id.

        Attributes:
            connector_id (str):
            created_at (datetime.datetime):
            forward (ConnectorEventForward): Which deliveries a destination is sent. unhandled: the ones the router acts on
                in no way, such as a Slack button click, a reaction or a modal submission, and a message no agent of the app
                answers: the app's own code next to the router's agent. all: every verified delivery, messages and grant events
                included, but the provider's URL handshake: the app runs its own agent. Either way a message an agent of the app
                answers is still answered there. A Slack reply that does not mention the bot, in a thread the agent is not in
                yet, is unhandled when it arrives. If the mention that starts the thread arrives within 10 minutes, as when
                Slack retries the mention, the agent answers the reply too. The forward is not taken back, and no event says
                that the agent answered it.
            id (str):
            updated_at (datetime.datetime): When the secret was last rotated, or the destination made.
            url (str):
            previous_secret_until (datetime.datetime | Unset): Until when the secret the last rotation replaced still signs
                beside the current one. Absent when only one secret signs.
    """

    connector_id: str
    created_at: datetime.datetime
    forward: ConnectorEventForward
    id: str
    updated_at: datetime.datetime
    url: str
    previous_secret_until: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        connector_id = self.connector_id

        created_at = self.created_at.isoformat()

        forward = self.forward.value

        id = self.id

        updated_at = self.updated_at.isoformat()

        url = self.url

        previous_secret_until: str | Unset = UNSET
        if not isinstance(self.previous_secret_until, Unset):
            previous_secret_until = self.previous_secret_until.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "connector_id": connector_id,
                "created_at": created_at,
                "forward": forward,
                "id": id,
                "updated_at": updated_at,
                "url": url,
            }
        )
        if previous_secret_until is not UNSET:
            field_dict["previous_secret_until"] = previous_secret_until

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        connector_id = d.pop("connector_id")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        forward = ConnectorEventForward(d.pop("forward"))

        id = d.pop("id")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        url = d.pop("url")

        _previous_secret_until = d.pop("previous_secret_until", UNSET)
        previous_secret_until: datetime.datetime | Unset
        if isinstance(_previous_secret_until, Unset):
            previous_secret_until = UNSET
        else:
            previous_secret_until = datetime.datetime.fromisoformat(
                _previous_secret_until
            )

        connector_event_destination = cls(
            connector_id=connector_id,
            created_at=created_at,
            forward=forward,
            id=id,
            updated_at=updated_at,
            url=url,
            previous_secret_until=previous_secret_until,
        )

        connector_event_destination.additional_properties = d
        return connector_event_destination

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
