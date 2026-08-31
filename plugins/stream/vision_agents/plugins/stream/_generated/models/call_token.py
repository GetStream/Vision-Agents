from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="CallToken")


@_attrs_define
class CallToken:
    """
    Attributes:
        api_key (str): The Stream app the call is in, which the browser SDK joins against.
        token (str):
        user_id (str):
        user_name (str):
        call_id (str): The Stream call to join, which is not the id this call is held by here.
        call_type (str):
        expires_at (datetime.datetime):
    """

    api_key: str
    token: str
    user_id: str
    user_name: str
    call_id: str
    call_type: str
    expires_at: datetime.datetime
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        token = self.token

        user_id = self.user_id

        user_name = self.user_name

        call_id = self.call_id

        call_type = self.call_type

        expires_at = self.expires_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "api_key": api_key,
                "token": token,
                "user_id": user_id,
                "user_name": user_name,
                "call_id": call_id,
                "call_type": call_type,
                "expires_at": expires_at,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        api_key = d.pop("api_key")

        token = d.pop("token")

        user_id = d.pop("user_id")

        user_name = d.pop("user_name")

        call_id = d.pop("call_id")

        call_type = d.pop("call_type")

        expires_at = datetime.datetime.fromisoformat(d.pop("expires_at"))

        call_token = cls(
            api_key=api_key,
            token=token,
            user_id=user_id,
            user_name=user_name,
            call_id=call_id,
            call_type=call_type,
            expires_at=expires_at,
        )

        call_token.additional_properties = d
        return call_token

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
