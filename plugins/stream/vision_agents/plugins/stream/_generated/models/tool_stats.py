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
        avg_turns_before_use (float | None): Turns a session took before its first call, on average: 0 is a call on the
            first turn. Null on the row over every tool.
        definition_tokens (int | None): What the tool's whole definition, its name, description and schema, costs every
            turn that carries it, estimated at four characters a token. Null when no call recorded it.
        loading (str): How the latest call had the tool offered: full, summary or deferred. Empty when no call recorded
            it, as calls made before it was recorded and built-in tools.
        session_share (float | None): Sessions that called it, out of the window's sessions, from 0 to 1. Null when no
            session took a turn.
        sessions (int): Sessions that called it at least once.
        summary_tokens (int | None): What its definition costs offered as a summary. Null unless its loading is summary.
    """

    avg_ms: float | None
    avg_tokens: float
    calls: int
    failed: int
    p99_ms: float | None
    tool: str
    avg_turns_before_use: float | None
    definition_tokens: int | None
    loading: str
    session_share: float | None
    sessions: int
    summary_tokens: int | None
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

        avg_turns_before_use: float | None
        avg_turns_before_use = self.avg_turns_before_use

        definition_tokens: int | None
        definition_tokens = self.definition_tokens

        loading = self.loading

        session_share: float | None
        session_share = self.session_share

        sessions = self.sessions

        summary_tokens: int | None
        summary_tokens = self.summary_tokens

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
                "avg_turns_before_use": avg_turns_before_use,
                "definition_tokens": definition_tokens,
                "loading": loading,
                "session_share": session_share,
                "sessions": sessions,
                "summary_tokens": summary_tokens,
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

        def _parse_avg_turns_before_use(data: object) -> float | None:
            if data is None:
                return data
            return cast(float | None, data)

        avg_turns_before_use = _parse_avg_turns_before_use(
            d.pop("avg_turns_before_use")
        )

        def _parse_definition_tokens(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        definition_tokens = _parse_definition_tokens(d.pop("definition_tokens"))

        loading = d.pop("loading")

        def _parse_session_share(data: object) -> float | None:
            if data is None:
                return data
            return cast(float | None, data)

        session_share = _parse_session_share(d.pop("session_share"))

        sessions = d.pop("sessions")

        def _parse_summary_tokens(data: object) -> int | None:
            if data is None:
                return data
            return cast(int | None, data)

        summary_tokens = _parse_summary_tokens(d.pop("summary_tokens"))

        tool_stats = cls(
            avg_ms=avg_ms,
            avg_tokens=avg_tokens,
            calls=calls,
            failed=failed,
            p99_ms=p99_ms,
            tool=tool,
            avg_turns_before_use=avg_turns_before_use,
            definition_tokens=definition_tokens,
            loading=loading,
            session_share=session_share,
            sessions=sessions,
            summary_tokens=summary_tokens,
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
