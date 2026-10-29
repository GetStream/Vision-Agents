from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="DataPolicy")


@_attrs_define
class DataPolicy:
    """What a caller requires of what happens to what they send: the audio they had transcribed, or the text they had
    spoken and the voice speaking it, and the region it is done in. This is a requirement rather than a description: a
    request naming one is only routed to a model whose declared handling meets it, and if none does the request is
    refused rather than sent somewhere that does not.

        Attributes:
            allow_training (bool | Unset): False requires a provider that has said it does not train on what it is sent.
                Omitting this asks nothing. A provider that has published nothing either way counts as not having said no.
            region (str | Unset): Where this call's work is done: us or eu for now. Only a model instance running in that
                region serves the request, and with none there it is refused (400), never served in another region. Omitted, the
                work is done in us. An agent's work is done in the agent's own region (agent.yaml's region), and a request
                naming another one for it is refused, naming both. A policy has no region. Example: eu.
            retention (str | Unset): The longest a provider may keep this audio - none, or a duration such as 30d or 24h.
                Omitting it asks nothing. Example: none.
    """

    allow_training: bool | Unset = UNSET
    region: str | Unset = UNSET
    retention: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allow_training = self.allow_training

        region = self.region

        retention = self.retention

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allow_training is not UNSET:
            field_dict["allow_training"] = allow_training
        if region is not UNSET:
            field_dict["region"] = region
        if retention is not UNSET:
            field_dict["retention"] = retention

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allow_training = d.pop("allow_training", UNSET)

        region = d.pop("region", UNSET)

        retention = d.pop("retention", UNSET)

        data_policy = cls(
            allow_training=allow_training,
            region=region,
            retention=retention,
        )

        data_policy.additional_properties = d
        return data_policy

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
