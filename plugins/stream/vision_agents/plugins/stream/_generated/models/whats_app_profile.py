from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="WhatsAppProfile")


@_attrs_define
class WhatsAppProfile:
    """
    Attributes:
        business_account (str | Unset): The WhatsApp Business account, existing or to create.
        business_portfolio (str | Unset): The Meta business portfolio, existing or to create.
        display_name (str | Unset):
        phone_number (str | Unset):
        verification_method (str | Unset): sms or voice.
    """

    business_account: str | Unset = UNSET
    business_portfolio: str | Unset = UNSET
    display_name: str | Unset = UNSET
    phone_number: str | Unset = UNSET
    verification_method: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        business_account = self.business_account

        business_portfolio = self.business_portfolio

        display_name = self.display_name

        phone_number = self.phone_number

        verification_method = self.verification_method

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if business_account is not UNSET:
            field_dict["business_account"] = business_account
        if business_portfolio is not UNSET:
            field_dict["business_portfolio"] = business_portfolio
        if display_name is not UNSET:
            field_dict["display_name"] = display_name
        if phone_number is not UNSET:
            field_dict["phone_number"] = phone_number
        if verification_method is not UNSET:
            field_dict["verification_method"] = verification_method

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        business_account = d.pop("business_account", UNSET)

        business_portfolio = d.pop("business_portfolio", UNSET)

        display_name = d.pop("display_name", UNSET)

        phone_number = d.pop("phone_number", UNSET)

        verification_method = d.pop("verification_method", UNSET)

        whats_app_profile = cls(
            business_account=business_account,
            business_portfolio=business_portfolio,
            display_name=display_name,
            phone_number=phone_number,
            verification_method=verification_method,
        )

        whats_app_profile.additional_properties = d
        return whats_app_profile

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
