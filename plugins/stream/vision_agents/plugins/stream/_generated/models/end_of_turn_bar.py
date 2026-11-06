from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="EndOfTurnBar")


@_attrs_define
class EndOfTurnBar:
    """What chorus-asr's p_eot must reach for a turn to have ended, at a check from from_ms into the pause on.

    Attributes:
        bar (float): What p_eot, chorus-asr's probability that the caller has finished, must reach for the turn to have
            ended: strictly between 0 and 1. Example: 0.9.
        from_ms (int): How far into the pause, in milliseconds, the bar begins. Example: 0.
    """

    bar: float
    from_ms: int

    def to_dict(self) -> dict[str, Any]:
        bar = self.bar

        from_ms = self.from_ms

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "bar": bar,
                "from_ms": from_ms,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        bar = d.pop("bar")

        from_ms = d.pop("from_ms")

        end_of_turn_bar = cls(
            bar=bar,
            from_ms=from_ms,
        )

        return end_of_turn_bar
