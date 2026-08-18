from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.phone_capability import PhoneCapability
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.phone_number_tags import PhoneNumberTags


T = TypeVar("T", bound="PhoneNumber")


@_attrs_define
class PhoneNumber:
    """
    Attributes:
        e164 (str):
        vendor (str):
        country (str):
        capabilities (list[PhoneCapability]):
        monthly_cost_micros (int):
        purchased_at (datetime.datetime):
        tags (PhoneNumberTags | Unset): The customer's own cost labels.
        stream_trunk_id (str | Unset): The SIP trunk calls to this number arrive on. Absent until attached.
        released_at (datetime.datetime | None | Unset):
    """

    e164: str
    vendor: str
    country: str
    capabilities: list[PhoneCapability]
    monthly_cost_micros: int
    purchased_at: datetime.datetime
    tags: PhoneNumberTags | Unset = UNSET
    stream_trunk_id: str | Unset = UNSET
    released_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        e164 = self.e164

        vendor = self.vendor

        country = self.country

        capabilities = []
        for capabilities_item_data in self.capabilities:
            capabilities_item = capabilities_item_data.value
            capabilities.append(capabilities_item)

        monthly_cost_micros = self.monthly_cost_micros

        purchased_at = self.purchased_at.isoformat()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        stream_trunk_id = self.stream_trunk_id

        released_at: None | str | Unset
        if isinstance(self.released_at, Unset):
            released_at = UNSET
        elif isinstance(self.released_at, datetime.datetime):
            released_at = self.released_at.isoformat()
        else:
            released_at = self.released_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "e164": e164,
                "vendor": vendor,
                "country": country,
                "capabilities": capabilities,
                "monthly_cost_micros": monthly_cost_micros,
                "purchased_at": purchased_at,
            }
        )
        if tags is not UNSET:
            field_dict["tags"] = tags
        if stream_trunk_id is not UNSET:
            field_dict["stream_trunk_id"] = stream_trunk_id
        if released_at is not UNSET:
            field_dict["released_at"] = released_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.phone_number_tags import PhoneNumberTags

        d = dict(src_dict)
        e164 = d.pop("e164")

        vendor = d.pop("vendor")

        country = d.pop("country")

        capabilities = []
        _capabilities = d.pop("capabilities")
        for capabilities_item_data in _capabilities:
            capabilities_item = PhoneCapability(capabilities_item_data)

            capabilities.append(capabilities_item)

        monthly_cost_micros = d.pop("monthly_cost_micros")

        purchased_at = datetime.datetime.fromisoformat(d.pop("purchased_at"))

        _tags = d.pop("tags", UNSET)
        tags: PhoneNumberTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = PhoneNumberTags.from_dict(_tags)

        stream_trunk_id = d.pop("stream_trunk_id", UNSET)

        def _parse_released_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                released_at_type_0 = datetime.datetime.fromisoformat(data)

                return released_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        released_at = _parse_released_at(d.pop("released_at", UNSET))

        phone_number = cls(
            e164=e164,
            vendor=vendor,
            country=country,
            capabilities=capabilities,
            monthly_cost_micros=monthly_cost_micros,
            purchased_at=purchased_at,
            tags=tags,
            stream_trunk_id=stream_trunk_id,
            released_at=released_at,
        )

        phone_number.additional_properties = d
        return phone_number

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
