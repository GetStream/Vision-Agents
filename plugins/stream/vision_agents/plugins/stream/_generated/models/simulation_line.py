from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SimulationLine")


@_attrs_define
class SimulationLine:
    """
    Attributes:
        caller (bool): True when the simulated caller said it rather than the agent.
        text (str):
        intended (str | Unset): What the agent meant to say, where that differs from what the caller heard. Only an
            audio simulation has both, and the difference is what running one is for.
        at (datetime.datetime | Unset):
    """

    caller: bool
    text: str
    intended: str | Unset = UNSET
    at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        caller = self.caller

        text = self.text

        intended = self.intended

        at: str | Unset = UNSET
        if not isinstance(self.at, Unset):
            at = self.at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "caller": caller,
                "text": text,
            }
        )
        if intended is not UNSET:
            field_dict["intended"] = intended
        if at is not UNSET:
            field_dict["at"] = at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        caller = d.pop("caller")

        text = d.pop("text")

        intended = d.pop("intended", UNSET)

        _at = d.pop("at", UNSET)
        at: datetime.datetime | Unset
        if isinstance(_at, Unset):
            at = UNSET
        else:
            at = datetime.datetime.fromisoformat(_at)

        simulation_line = cls(
            caller=caller,
            text=text,
            intended=intended,
            at=at,
        )

        simulation_line.additional_properties = d
        return simulation_line

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
