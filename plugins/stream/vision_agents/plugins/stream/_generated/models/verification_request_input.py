from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.verification_request_input_calls_per_month import (
    VerificationRequestInputCallsPerMonth,
)
from ..models.verification_request_input_country import VerificationRequestInputCountry

T = TypeVar("T", bound="VerificationRequestInput")


@_attrs_define
class VerificationRequestInput:
    """
    Attributes:
        calls_per_month (VerificationRequestInputCallsPerMonth): How many calls a month the organization expects.
        country (VerificationRequestInputCountry): Where the organization is registered: US or CA.
        email (str): Where Stream answers.
        legal_name (str): The organization's registered legal name.
        tax_id (str): The organization's tax id: an EIN in the US, a BN in Canada. Not checked further.
        use_case (str): What the organization's agents will do on calls.
        website (str): The organization's website, an http or https URL.
    """

    calls_per_month: VerificationRequestInputCallsPerMonth
    country: VerificationRequestInputCountry
    email: str
    legal_name: str
    tax_id: str
    use_case: str
    website: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        calls_per_month = self.calls_per_month.value

        country = self.country.value

        email = self.email

        legal_name = self.legal_name

        tax_id = self.tax_id

        use_case = self.use_case

        website = self.website

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "calls_per_month": calls_per_month,
                "country": country,
                "email": email,
                "legal_name": legal_name,
                "tax_id": tax_id,
                "use_case": use_case,
                "website": website,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        calls_per_month = VerificationRequestInputCallsPerMonth._decode(
            d.pop("calls_per_month")
        )

        country = VerificationRequestInputCountry._decode(d.pop("country"))

        email = d.pop("email")

        legal_name = d.pop("legal_name")

        tax_id = d.pop("tax_id")

        use_case = d.pop("use_case")

        website = d.pop("website")

        verification_request_input = cls(
            calls_per_month=calls_per_month,
            country=country,
            email=email,
            legal_name=legal_name,
            tax_id=tax_id,
            use_case=use_case,
            website=website,
        )

        verification_request_input.additional_properties = d
        return verification_request_input

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
