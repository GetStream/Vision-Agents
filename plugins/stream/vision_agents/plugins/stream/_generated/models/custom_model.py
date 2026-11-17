from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomModel")


@_attrs_define
class CustomModel:
    """
    Attributes:
        base_url (str):
        context_window (int):
        created_at (datetime.datetime):
        has_api_key (bool): Whether a key is stored. The key itself is never returned.
        id (str):
        input_modalities (list[str]):
        model (str):
        name (str):
        per_million_input_tokens (float):
        per_million_output_tokens (float):
        target (str): What a router config or session names the model by. Example: custom/support-qwen.
        updated_at (datetime.datetime):
        retention (str | Unset):
        trains_on_data (str | Unset):
    """

    base_url: str
    context_window: int
    created_at: datetime.datetime
    has_api_key: bool
    id: str
    input_modalities: list[str]
    model: str
    name: str
    per_million_input_tokens: float
    per_million_output_tokens: float
    target: str
    updated_at: datetime.datetime
    retention: str | Unset = UNSET
    trains_on_data: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base_url = self.base_url

        context_window = self.context_window

        created_at = self.created_at.isoformat()

        has_api_key = self.has_api_key

        id = self.id

        input_modalities = self.input_modalities

        model = self.model

        name = self.name

        per_million_input_tokens = self.per_million_input_tokens

        per_million_output_tokens = self.per_million_output_tokens

        target = self.target

        updated_at = self.updated_at.isoformat()

        retention = self.retention

        trains_on_data = self.trains_on_data

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "base_url": base_url,
                "context_window": context_window,
                "created_at": created_at,
                "has_api_key": has_api_key,
                "id": id,
                "input_modalities": input_modalities,
                "model": model,
                "name": name,
                "per_million_input_tokens": per_million_input_tokens,
                "per_million_output_tokens": per_million_output_tokens,
                "target": target,
                "updated_at": updated_at,
            }
        )
        if retention is not UNSET:
            field_dict["retention"] = retention
        if trains_on_data is not UNSET:
            field_dict["trains_on_data"] = trains_on_data

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        base_url = d.pop("base_url")

        context_window = d.pop("context_window")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        has_api_key = d.pop("has_api_key")

        id = d.pop("id")

        input_modalities = cast(list[str], d.pop("input_modalities"))

        model = d.pop("model")

        name = d.pop("name")

        per_million_input_tokens = d.pop("per_million_input_tokens")

        per_million_output_tokens = d.pop("per_million_output_tokens")

        target = d.pop("target")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        retention = d.pop("retention", UNSET)

        trains_on_data = d.pop("trains_on_data", UNSET)

        custom_model = cls(
            base_url=base_url,
            context_window=context_window,
            created_at=created_at,
            has_api_key=has_api_key,
            id=id,
            input_modalities=input_modalities,
            model=model,
            name=name,
            per_million_input_tokens=per_million_input_tokens,
            per_million_output_tokens=per_million_output_tokens,
            target=target,
            updated_at=updated_at,
            retention=retention,
            trains_on_data=trains_on_data,
        )

        custom_model.additional_properties = d
        return custom_model

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
