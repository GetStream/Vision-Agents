from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.end_of_turn_bar import EndOfTurnBar


T = TypeVar("T", bound="EndOfTurn")


@_attrs_define
class EndOfTurn:
    """When chorus-asr has a caller's turn ended, where the router asks it about turns: one of max_wait_ms, preset and
    bars. An empty object is the server's default preset. Whichever is named replaces only the bars: when a pause is
    checked, how much more a turn's first pause asks, and the pause that ends a turn whatever the model says stay the
    server's. What a preset or a wait means is the policy of the model the server runs, read when a pause is checked;
    one that model no longer publishes ends turns by its default.

        Attributes:
            bars (list[EndOfTurnBar] | Unset): Bars of the agent's own: the first from 0 ms, each later one from further
                into the pause. At each check of a pause, the turn has ended when p_eot reaches the last bar to have begun.
            max_wait_ms (int | Unset): The most a caller who has finished may wait for the answer, in milliseconds. It
                chooses the point of chorus-asr's measured trade-off with the fewest cut-ins whose measured p90 wait is within
                it. Refused when it is shorter than every point's. Example: 900.
            preset (str | Unset): One of the presets the chorus-asr server publishes, by name, such as safe or fast. Refused
                when the server publishes none by that name. Example: safe.
    """

    bars: list[EndOfTurnBar] | Unset = UNSET
    max_wait_ms: int | Unset = UNSET
    preset: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        bars: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.bars, Unset):
            bars = []
            for bars_item_data in self.bars:
                bars_item = bars_item_data.to_dict()
                bars.append(bars_item)

        max_wait_ms = self.max_wait_ms

        preset = self.preset

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if bars is not UNSET:
            field_dict["bars"] = bars
        if max_wait_ms is not UNSET:
            field_dict["max_wait_ms"] = max_wait_ms
        if preset is not UNSET:
            field_dict["preset"] = preset

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.end_of_turn_bar import EndOfTurnBar

        d = dict(src_dict)
        _bars = d.pop("bars", UNSET)
        bars: list[EndOfTurnBar] | Unset = UNSET
        if _bars is not UNSET:
            bars = []
            for bars_item_data in _bars:
                bars_item = EndOfTurnBar.from_dict(bars_item_data)

                bars.append(bars_item)

        max_wait_ms = d.pop("max_wait_ms", UNSET)

        preset = d.pop("preset", UNSET)

        end_of_turn = cls(
            bars=bars,
            max_wait_ms=max_wait_ms,
            preset=preset,
        )

        return end_of_turn
