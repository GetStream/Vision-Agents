from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="ToolStats")


@_attrs_define
class ToolStats:
    """
    Attributes:
        avg_ms (float | None): From the call starting to its result. Null when none was timed.
        avg_tokens (float): What a call put in front of the model, its arguments and its result, estimated at four
            characters a token.
        calls (int):
        failed (int): Calls that ended in an error.
        p99_ms (float | None):
        tool (str):
    """

    avg_ms: float | None
    avg_tokens: float
    calls: int
    failed: int
    p99_ms: float | None
    tool: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        avg_ms: float | None
        avg_ms = self.avg_ms

        avg_tokens = self.avg_tokens

        calls = self.calls

        failed = self.failed

        p99_ms: float | None
        p99_ms = self.p99_ms

        tool = self.tool

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "avg_ms": avg_ms,
                "avg_tokens": avg_tokens,
                "calls": calls,
                "failed": failed,
                "p99_ms": p99_ms,
                "tool": tool,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)

        def _parse_avg_ms(data: object) -> float | None:
            if data is None:
                return data
            return cast(float | None, data)

        avg_ms = _parse_avg_ms(d.pop("avg_ms"))

        avg_tokens = d.pop("avg_tokens")

        calls = d.pop("calls")

        failed = d.pop("failed")

        def _parse_p99_ms(data: object) -> float | None:
            if data is None:
                return data
            return cast(float | None, data)

        p99_ms = _parse_p99_ms(d.pop("p99_ms"))

        tool = d.pop("tool")

        tool_stats = cls(
            avg_ms=avg_ms,
            avg_tokens=avg_tokens,
            calls=calls,
            failed=failed,
            p99_ms=p99_ms,
            tool=tool,
        )

        tool_stats.additional_properties = d
        return tool_stats

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
