from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.transfer_destination import TransferDestination


T = TypeVar("T", bound="TransferSettings")


@_attrs_define
class TransferSettings:
    """
    Attributes:
        destinations (list[TransferDestination] | None | Unset): The numbers the agent may transfer to, by name. Names
            are unique ignoring case and surrounding spaces.
        enabled (bool | Unset): Offer the tool. Omitted is on.
    """

    destinations: list[TransferDestination] | None | Unset = UNSET
    enabled: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        destinations: list[dict[str, Any]] | None | Unset
        if isinstance(self.destinations, Unset):
            destinations = UNSET
        elif isinstance(self.destinations, list):
            destinations = []
            for destinations_type_0_item_data in self.destinations:
                destinations_type_0_item = destinations_type_0_item_data.to_dict()
                destinations.append(destinations_type_0_item)

        else:
            destinations = self.destinations

        enabled = self.enabled

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if destinations is not UNSET:
            field_dict["destinations"] = destinations
        if enabled is not UNSET:
            field_dict["enabled"] = enabled

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.transfer_destination import TransferDestination

        d = dict(src_dict)

        def _parse_destinations(
            data: object,
        ) -> list[TransferDestination] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                destinations_type_0 = []
                _destinations_type_0 = data
                for destinations_type_0_item_data in _destinations_type_0:
                    destinations_type_0_item = TransferDestination.from_dict(
                        destinations_type_0_item_data
                    )

                    destinations_type_0.append(destinations_type_0_item)

                return destinations_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[TransferDestination] | None | Unset, data)

        destinations = _parse_destinations(d.pop("destinations", UNSET))

        enabled = d.pop("enabled", UNSET)

        transfer_settings = cls(
            destinations=destinations,
            enabled=enabled,
        )

        transfer_settings.additional_properties = d
        return transfer_settings

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
