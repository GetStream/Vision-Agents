from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="StsOptionsOverwrites")


@_attrs_define
class StsOptionsOverwrites:
    """Settings for one provider that this vocabulary has no word for, keyed by provider name, for example {"openai":
    {"eagerness": "high"}}. The provider named parses its own block and refuses a field it does not have, so an
    overwrite is either sent or reported rather than accepted and dropped.

        Example:
            {'openai': {'eagerness': 'high'}}

    """

    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        sts_options_overwrites = cls()

        sts_options_overwrites.additional_properties = d
        return sts_options_overwrites

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
