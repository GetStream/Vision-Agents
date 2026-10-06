from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.create_opt_out_request_source import CreateOptOutRequestSource
from ..models.opt_out_channel import OptOutChannel
from ..types import UNSET, Unset

T = TypeVar("T", bound="CreateOptOutRequest")


@_attrs_define
class CreateOptOutRequest:
    """
    Attributes:
        channel (OptOutChannel): The channel a recipient opted out of, or all of them.
        recipient (str): The number, in E.164.
        source (CreateOptOutRequestSource | Unset): Omitted is api.
    """

    channel: OptOutChannel
    recipient: str
    source: CreateOptOutRequestSource | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        channel = self.channel.value

        recipient = self.recipient

        source: str | Unset = UNSET
        if not isinstance(self.source, Unset):
            source = self.source.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "channel": channel,
                "recipient": recipient,
            }
        )
        if source is not UNSET:
            field_dict["source"] = source

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        channel = OptOutChannel(d.pop("channel"))

        recipient = d.pop("recipient")

        _source = d.pop("source", UNSET)
        source: CreateOptOutRequestSource | Unset
        if isinstance(_source, Unset):
            source = UNSET
        else:
            source = CreateOptOutRequestSource(_source)

        create_opt_out_request = cls(
            channel=channel,
            recipient=recipient,
            source=source,
        )

        create_opt_out_request.additional_properties = d
        return create_opt_out_request

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
