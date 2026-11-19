from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="FloorRuling")


@_attrs_define
class FloorRuling:
    """
    Attributes:
        disposition (str): respond, wait, clarify or ignore.
        floor (str): stop, shorten or continue over the agent; at a pause, who has the floor once the turn has ended.
        cannot (bool | Unset): The caller wants something none of the agent's tools can do.
        tool (str | Unset): The tool the caller wants.
    """

    disposition: str
    floor: str
    cannot: bool | Unset = UNSET
    tool: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        disposition = self.disposition

        floor = self.floor

        cannot = self.cannot

        tool = self.tool

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "disposition": disposition,
                "floor": floor,
            }
        )
        if cannot is not UNSET:
            field_dict["cannot"] = cannot
        if tool is not UNSET:
            field_dict["tool"] = tool

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        disposition = d.pop("disposition")

        floor = d.pop("floor")

        cannot = d.pop("cannot", UNSET)

        tool = d.pop("tool", UNSET)

        floor_ruling = cls(
            disposition=disposition,
            floor=floor,
            cannot=cannot,
            tool=tool,
        )

        floor_ruling.additional_properties = d
        return floor_ruling

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
