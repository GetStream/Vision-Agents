from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.verification_request_calls_per_month import (
    VerificationRequestCallsPerMonth,
)
from ..models.verification_request_country import VerificationRequestCountry

T = TypeVar("T", bound="VerificationRequest")


@_attrs_define
class VerificationRequest:
    """An organization's request to be verified: who it is, what its agents will do and where Stream answers. An
    organization has one, replaced whenever it sends again.

        Attributes:
            calls_per_month (VerificationRequestCallsPerMonth): How many calls a month the organization expects.
            country (VerificationRequestCountry): Where the organization is registered: US or CA.
            email (str): Where Stream answers.
            legal_name (str): The organization's registered legal name.
            submitted_at (datetime.datetime): When the router took the request.
            submitted_by (str): The user the request was made as, when the caller said, else empty.
            tax_id (str): The organization's tax id: an EIN in the US, a BN in Canada. Not checked further.
            use_case (str): What the organization's agents will do on calls.
            website (str): The organization's website, an http or https URL.
    """

    calls_per_month: VerificationRequestCallsPerMonth
    country: VerificationRequestCountry
    email: str
    legal_name: str
    submitted_at: datetime.datetime
    submitted_by: str
    tax_id: str
    use_case: str
    website: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        calls_per_month = self.calls_per_month.value

        country = self.country.value

        email = self.email

        legal_name = self.legal_name

        submitted_at = self.submitted_at.isoformat()

        submitted_by = self.submitted_by

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
                "submitted_at": submitted_at,
                "submitted_by": submitted_by,
                "tax_id": tax_id,
                "use_case": use_case,
                "website": website,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        calls_per_month = VerificationRequestCallsPerMonth._decode(
            d.pop("calls_per_month")
        )

        country = VerificationRequestCountry._decode(d.pop("country"))

        email = d.pop("email")

        legal_name = d.pop("legal_name")

        submitted_at = datetime.datetime.fromisoformat(d.pop("submitted_at"))

        submitted_by = d.pop("submitted_by")

        tax_id = d.pop("tax_id")

        use_case = d.pop("use_case")

        website = d.pop("website")

        verification_request = cls(
            calls_per_month=calls_per_month,
            country=country,
            email=email,
            legal_name=legal_name,
            submitted_at=submitted_at,
            submitted_by=submitted_by,
            tax_id=tax_id,
            use_case=use_case,
            website=website,
        )

        verification_request.additional_properties = d
        return verification_request

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
