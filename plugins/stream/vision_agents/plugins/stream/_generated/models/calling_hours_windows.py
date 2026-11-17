from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.calling_window import CallingWindow


T = TypeVar("T", bound="CallingHoursWindows")


@_attrs_define
class CallingHoursWindows:
    """Keyed monday to sunday. A day left out rings nobody."""

    additional_properties: dict[str, list[CallingWindow] | None] = _attrs_field(
        init=False, factory=dict
    )

    def to_dict(self) -> dict[str, Any]:

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            if isinstance(prop, list):
                field_dict[prop_name] = []
                for additional_property_type_0_item_data in prop:
                    additional_property_type_0_item = (
                        additional_property_type_0_item_data.to_dict()
                    )
                    field_dict[prop_name].append(additional_property_type_0_item)

            else:
                field_dict[prop_name] = prop

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.calling_window import CallingWindow

        d = dict(src_dict)
        calling_hours_windows = cls()

        additional_properties = {}
        for prop_name, prop_dict in d.items():

            def _parse_additional_property(data: object) -> list[CallingWindow] | None:
                if data is None:
                    return data
                try:
                    if not isinstance(data, list):
                        raise TypeError()
                    additional_property_type_0 = []
                    _additional_property_type_0 = data
                    for (
                        additional_property_type_0_item_data
                    ) in _additional_property_type_0:
                        additional_property_type_0_item = CallingWindow.from_dict(
                            additional_property_type_0_item_data
                        )

                        additional_property_type_0.append(
                            additional_property_type_0_item
                        )

                    return additional_property_type_0
                except (TypeError, ValueError, AttributeError, KeyError):
                    pass
                return cast(list[CallingWindow] | None, data)

            additional_property = _parse_additional_property(prop_dict)

            additional_properties[prop_name] = additional_property

        calling_hours_windows.additional_properties = additional_properties
        return calling_hours_windows

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> list[CallingWindow] | None:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: list[CallingWindow] | None) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
