from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.opt_out_channel import OptOutChannel

T = TypeVar("T", bound="OptOut")


@_attrs_define
class OptOut:
    """Somebody who asked not to be reached. Nothing is texted or dialled to them on the channel, or on any for all, until
    the opt-out is revoked or they text START.

        Attributes:
            channel (OptOutChannel): The channel a recipient opted out of, or all of them.
            created_at (datetime.datetime):
            id (str):
            recipient (str): The number, in E.164.
            source (str): keyword when they texted STOP, otherwise api or dashboard.
    """

    channel: OptOutChannel
    created_at: datetime.datetime
    id: str
    recipient: str
    source: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        channel = self.channel.value

        created_at = self.created_at.isoformat()

        id = self.id

        recipient = self.recipient

        source = self.source

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "channel": channel,
                "created_at": created_at,
                "id": id,
                "recipient": recipient,
                "source": source,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        channel = OptOutChannel(d.pop("channel"))

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        recipient = d.pop("recipient")

        source = d.pop("source")

        opt_out = cls(
            channel=channel,
            created_at=created_at,
            id=id,
            recipient=recipient,
            source=source,
        )

        opt_out.additional_properties = d
        return opt_out

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
