from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.verification_request import VerificationRequest


T = TypeVar("T", bound="OrganizationVerificationRequest")


@_attrs_define
class OrganizationVerificationRequest:
    """
    Attributes:
        request (None | VerificationRequest): The organization's request, null when it never sent one.
    """

    request: None | VerificationRequest
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.verification_request import VerificationRequest

        request: dict[str, Any] | None
        if isinstance(self.request, VerificationRequest):
            request = self.request.to_dict()
        else:
            request = self.request

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "request": request,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.verification_request import VerificationRequest

        d = dict(src_dict)

        def _parse_request(data: object) -> None | VerificationRequest:
            if data is None:
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                request_type_0 = VerificationRequest.from_dict(data)

                return request_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | VerificationRequest, data)

        request = _parse_request(d.pop("request"))

        organization_verification_request = cls(
            request=request,
        )

        organization_verification_request.additional_properties = d
        return organization_verification_request

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
