from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="TransferDestination")


@_attrs_define
class TransferDestination:
    """
    Attributes:
        name (str): What the agent calls this destination, such as Sales.
        number (str): The number to transfer to, in E.164 form.
        warm (bool | Unset): Announce the caller to the person answering before connecting them.
        when (str | Unset): When the agent should transfer here, in the caller's terms.
    """

    name: str
    number: str
    warm: bool | Unset = UNSET
    when: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        number = self.number

        warm = self.warm

        when = self.when

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "name": name,
                "number": number,
            }
        )
        if warm is not UNSET:
            field_dict["warm"] = warm
        if when is not UNSET:
            field_dict["when"] = when

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        name = d.pop("name")

        number = d.pop("number")

        warm = d.pop("warm", UNSET)

        when = d.pop("when", UNSET)

        transfer_destination = cls(
            name=name,
            number=number,
            warm=warm,
            when=when,
        )

        transfer_destination.additional_properties = d
        return transfer_destination

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
