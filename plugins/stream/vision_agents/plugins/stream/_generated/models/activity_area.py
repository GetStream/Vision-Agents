from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ActivityArea")


@_attrs_define
class ActivityArea:
    """
    Attributes:
        area (str): The area of the API, such as agent_setup or conversations.
        client (str): The client the requests named themselves as: dashboard, cli, sdk or api.
        first_at (datetime.datetime): When the first request arrived.
        last_at (datetime.datetime): When the last request arrived.
        last_refusal_code (str): The error code of the latest refusal, empty when there was none.
        refused (int): How many were answered with a status of 400 or more.
        requests (int): How many requests arrived, refused ones included.
    """

    area: str
    client: str
    first_at: datetime.datetime
    last_at: datetime.datetime
    last_refusal_code: str
    refused: int
    requests: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        area = self.area

        client = self.client

        first_at = self.first_at.isoformat()

        last_at = self.last_at.isoformat()

        last_refusal_code = self.last_refusal_code

        refused = self.refused

        requests = self.requests

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "area": area,
                "client": client,
                "first_at": first_at,
                "last_at": last_at,
                "last_refusal_code": last_refusal_code,
                "refused": refused,
                "requests": requests,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        area = d.pop("area")

        client = d.pop("client")

        first_at = datetime.datetime.fromisoformat(d.pop("first_at"))

        last_at = datetime.datetime.fromisoformat(d.pop("last_at"))

        last_refusal_code = d.pop("last_refusal_code")

        refused = d.pop("refused")

        requests = d.pop("requests")

        activity_area = cls(
            area=area,
            client=client,
            first_at=first_at,
            last_at=last_at,
            last_refusal_code=last_refusal_code,
            refused=refused,
            requests=requests,
        )

        activity_area.additional_properties = d
        return activity_area

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
