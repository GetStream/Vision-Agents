from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.verification_request import VerificationRequest


T = TypeVar("T", bound="VerificationRequestEntry")


@_attrs_define
class VerificationRequestEntry:
    """
    Attributes:
        organization_id (str):
        request (VerificationRequest): An organization's request to be verified: who it is, what its agents will do and
            where Stream answers. An organization has one, replaced whenever it sends again.
        tier (str): The organization's last tier the router knows: verified or not-verified, empty when unknown.
    """

    organization_id: str
    request: VerificationRequest
    tier: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        organization_id = self.organization_id

        request = self.request.to_dict()

        tier = self.tier

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "organization_id": organization_id,
                "request": request,
                "tier": tier,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.verification_request import VerificationRequest

        d = dict(src_dict)
        organization_id = d.pop("organization_id")

        request = VerificationRequest.from_dict(d.pop("request"))

        tier = d.pop("tier")

        verification_request_entry = cls(
            organization_id=organization_id,
            request=request,
            tier=tier,
        )

        verification_request_entry.additional_properties = d
        return verification_request_entry

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
