from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="RespondRequest")


@_attrs_define
class RespondRequest:
    """
    Attributes:
        text (str):
        client_id (str | Unset): The install the command came from. It is written on the person's message as client_id,
            and a client tool called while answering is addressed to it.
        request_id (str | Unset): Generated and sent by the SDKs, one per question, so a retry is answered once.
            Required for personal persistent text conversations, and ignored by a session not kept in Stream Chat. A retry
            with the same id and text does not restart inference.
    """

    text: str
    client_id: str | Unset = UNSET
    request_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        text = self.text

        client_id = self.client_id

        request_id = self.request_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "text": text,
            }
        )
        if client_id is not UNSET:
            field_dict["client_id"] = client_id
        if request_id is not UNSET:
            field_dict["request_id"] = request_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        text = d.pop("text")

        client_id = d.pop("client_id", UNSET)

        request_id = d.pop("request_id", UNSET)

        respond_request = cls(
            text=text,
            client_id=client_id,
            request_id=request_id,
        )

        respond_request.additional_properties = d
        return respond_request

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
