from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.custom_model_request_trains_on_data import CustomModelRequestTrainsOnData
from ..types import UNSET, Unset

T = TypeVar("T", bound="CustomModelRequest")


@_attrs_define
class CustomModelRequest:
    """
    Attributes:
        base_url (str): The endpoint root, up to and including /v1, that chat completions are posted under. Example:
            https://model-abc123.api.baseten.co/environments/production/sync/v1.
        model (str): The id the endpoint serves the weights under. Example: Qwen/Qwen3.8-27B.
        name (str): What a config names the model by, as `custom/<name>`. Unique among the customer's models. Example:
            support-qwen.
        api_key (str | Unset): Sent as a bearer token. Stored sealed and never returned. Left out of an update keeps the
            one stored; empty removes it.
        context_window (int | Unset): Tokens the model accepts, so the router can refuse a conversation that would not
            fit. Omitted is unknown.
        input_modalities (list[str] | Unset): Input kinds beyond text the model accepts, e.g. image.
        per_million_input_tokens (float | Unset): What the host bills per million input tokens in USD, so usage can say
            what a conversation cost. Omitted is free.
        per_million_output_tokens (float | Unset): What the host bills per million output tokens in USD. Omitted is
            free.
        retention (str | Unset): How long the host keeps what it is sent: none, or a duration such as 30d or 24h.
            Example: none.
        trains_on_data (CustomModelRequestTrainsOnData | Unset): Whether the host trains on what it is sent. Declared
            with retention, or a session with a data policy is never routed here.
    """

    base_url: str
    model: str
    name: str
    api_key: str | Unset = UNSET
    context_window: int | Unset = UNSET
    input_modalities: list[str] | Unset = UNSET
    per_million_input_tokens: float | Unset = UNSET
    per_million_output_tokens: float | Unset = UNSET
    retention: str | Unset = UNSET
    trains_on_data: CustomModelRequestTrainsOnData | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base_url = self.base_url

        model = self.model

        name = self.name

        api_key = self.api_key

        context_window = self.context_window

        input_modalities: list[str] | Unset = UNSET
        if not isinstance(self.input_modalities, Unset):
            input_modalities = self.input_modalities

        per_million_input_tokens = self.per_million_input_tokens

        per_million_output_tokens = self.per_million_output_tokens

        retention = self.retention

        trains_on_data: str | Unset = UNSET
        if not isinstance(self.trains_on_data, Unset):
            trains_on_data = self.trains_on_data.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "base_url": base_url,
                "model": model,
                "name": name,
            }
        )
        if api_key is not UNSET:
            field_dict["api_key"] = api_key
        if context_window is not UNSET:
            field_dict["context_window"] = context_window
        if input_modalities is not UNSET:
            field_dict["input_modalities"] = input_modalities
        if per_million_input_tokens is not UNSET:
            field_dict["per_million_input_tokens"] = per_million_input_tokens
        if per_million_output_tokens is not UNSET:
            field_dict["per_million_output_tokens"] = per_million_output_tokens
        if retention is not UNSET:
            field_dict["retention"] = retention
        if trains_on_data is not UNSET:
            field_dict["trains_on_data"] = trains_on_data

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        base_url = d.pop("base_url")

        model = d.pop("model")

        name = d.pop("name")

        api_key = d.pop("api_key", UNSET)

        context_window = d.pop("context_window", UNSET)

        input_modalities = cast(list[str], d.pop("input_modalities", UNSET))

        per_million_input_tokens = d.pop("per_million_input_tokens", UNSET)

        per_million_output_tokens = d.pop("per_million_output_tokens", UNSET)

        retention = d.pop("retention", UNSET)

        _trains_on_data = d.pop("trains_on_data", UNSET)
        trains_on_data: CustomModelRequestTrainsOnData | Unset
        if isinstance(_trains_on_data, Unset):
            trains_on_data = UNSET
        else:
            trains_on_data = CustomModelRequestTrainsOnData(_trains_on_data)

        custom_model_request = cls(
            base_url=base_url,
            model=model,
            name=name,
            api_key=api_key,
            context_window=context_window,
            input_modalities=input_modalities,
            per_million_input_tokens=per_million_input_tokens,
            per_million_output_tokens=per_million_output_tokens,
            retention=retention,
            trains_on_data=trains_on_data,
        )

        custom_model_request.additional_properties = d
        return custom_model_request

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
