from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ModelCallTiming")


@_attrs_define
class ModelCallTiming:
    """
    Attributes:
        started_at (datetime.datetime):
        provider (str):
        model (str):
        success (bool):
        operation_id (str | Unset): Response ID for this operation; retries can share an ID.
        purpose (str | Unset): reply, flow or subagent.
        ttft_ms (float | Unset): Request to first token.
        duration_ms (float | Unset): Request to completed response or failed create.
        input_tokens (int | Unset):
        output_tokens (int | Unset):
    """

    started_at: datetime.datetime
    provider: str
    model: str
    success: bool
    operation_id: str | Unset = UNSET
    purpose: str | Unset = UNSET
    ttft_ms: float | Unset = UNSET
    duration_ms: float | Unset = UNSET
    input_tokens: int | Unset = UNSET
    output_tokens: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        started_at = self.started_at.isoformat()

        provider = self.provider

        model = self.model

        success = self.success

        operation_id = self.operation_id

        purpose = self.purpose

        ttft_ms = self.ttft_ms

        duration_ms = self.duration_ms

        input_tokens = self.input_tokens

        output_tokens = self.output_tokens

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "started_at": started_at,
                "provider": provider,
                "model": model,
                "success": success,
            }
        )
        if operation_id is not UNSET:
            field_dict["operation_id"] = operation_id
        if purpose is not UNSET:
            field_dict["purpose"] = purpose
        if ttft_ms is not UNSET:
            field_dict["ttft_ms"] = ttft_ms
        if duration_ms is not UNSET:
            field_dict["duration_ms"] = duration_ms
        if input_tokens is not UNSET:
            field_dict["input_tokens"] = input_tokens
        if output_tokens is not UNSET:
            field_dict["output_tokens"] = output_tokens

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        started_at = datetime.datetime.fromisoformat(d.pop("started_at"))

        provider = d.pop("provider")

        model = d.pop("model")

        success = d.pop("success")

        operation_id = d.pop("operation_id", UNSET)

        purpose = d.pop("purpose", UNSET)

        ttft_ms = d.pop("ttft_ms", UNSET)

        duration_ms = d.pop("duration_ms", UNSET)

        input_tokens = d.pop("input_tokens", UNSET)

        output_tokens = d.pop("output_tokens", UNSET)

        model_call_timing = cls(
            started_at=started_at,
            provider=provider,
            model=model,
            success=success,
            operation_id=operation_id,
            purpose=purpose,
            ttft_ms=ttft_ms,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        model_call_timing.additional_properties = d
        return model_call_timing

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
