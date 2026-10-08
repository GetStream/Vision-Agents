from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="IMessageProfile")


@_attrs_define
class IMessageProfile:
    """
    Attributes:
        consent_method (str | Unset):
        contact_card_image_url (str | Unset): Square, at least 200 by 200.
        contact_card_name (str | Unset):
        fallback_channels (list[str] | None | Unset): imessage, rcs or sms, in the order to try them.
    """

    consent_method: str | Unset = UNSET
    contact_card_image_url: str | Unset = UNSET
    contact_card_name: str | Unset = UNSET
    fallback_channels: list[str] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        consent_method = self.consent_method

        contact_card_image_url = self.contact_card_image_url

        contact_card_name = self.contact_card_name

        fallback_channels: list[str] | None | Unset
        if isinstance(self.fallback_channels, Unset):
            fallback_channels = UNSET
        elif isinstance(self.fallback_channels, list):
            fallback_channels = self.fallback_channels

        else:
            fallback_channels = self.fallback_channels

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if consent_method is not UNSET:
            field_dict["consent_method"] = consent_method
        if contact_card_image_url is not UNSET:
            field_dict["contact_card_image_url"] = contact_card_image_url
        if contact_card_name is not UNSET:
            field_dict["contact_card_name"] = contact_card_name
        if fallback_channels is not UNSET:
            field_dict["fallback_channels"] = fallback_channels

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        consent_method = d.pop("consent_method", UNSET)

        contact_card_image_url = d.pop("contact_card_image_url", UNSET)

        contact_card_name = d.pop("contact_card_name", UNSET)

        def _parse_fallback_channels(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                fallback_channels_type_0 = cast(list[str], data)

                return fallback_channels_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        fallback_channels = _parse_fallback_channels(d.pop("fallback_channels", UNSET))

        i_message_profile = cls(
            consent_method=consent_method,
            contact_card_image_url=contact_card_image_url,
            contact_card_name=contact_card_name,
            fallback_channels=fallback_channels,
        )

        i_message_profile.additional_properties = d
        return i_message_profile

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
