from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.floor_check_kind import FloorCheckKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.floor_option import FloorOption
    from ..models.floor_ruling import FloorRuling


T = TypeVar("T", bound="FloorCheck")


@_attrs_define
class FloorCheck:
    """
    Attributes:
        acted (bool): Whether the router acted on the ruling: with the floor free, the caller's turn ended on this
            check; while the agent talked, its stop or shorten was handed to the agent, which may still talk on over its own
            echo or a murmur, or the turn ended on it. False for a ruling to go on, and for one the router let go of, as for
            words since replaced.
        at (datetime.datetime): When the check was asked.
        kind (FloorCheckKind): What a chorus-asr floor-control check asked: pause, with the floor free, whether the
            caller has finished; over_agent, while the agent was talking, whether the caller stops or shortens it, at a
            pause or while they talked on.
        latency_ms (float): The round trip to the chorus-asr server.
        ruling (FloorRuling):
        options (list[FloorOption] | Unset): The probability of each answer, in the order the model was offered them:
            respond, wait, clarify and ignore at a pause; stop, shorten, continue and ignore over the agent; then cannot and
            each of the agent's tools, those of at least 0.01. Absent from a server that does not report them.
        paused_ms (float | Unset): How far into the caller's pause the check was asked. Absent for a check asked while
            they talked on over the agent.
        point (str | Unset): The end-of-turn point or preset the check was read by. Absent over the agent, where there
            is no bar.
    """

    acted: bool
    at: datetime.datetime
    kind: FloorCheckKind
    latency_ms: float
    ruling: FloorRuling
    options: list[FloorOption] | Unset = UNSET
    paused_ms: float | Unset = UNSET
    point: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        acted = self.acted

        at = self.at.isoformat()

        kind = self.kind.value

        latency_ms = self.latency_ms

        ruling = self.ruling.to_dict()

        options: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = []
            for options_item_data in self.options:
                options_item = options_item_data.to_dict()
                options.append(options_item)

        paused_ms = self.paused_ms

        point = self.point

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "acted": acted,
                "at": at,
                "kind": kind,
                "latency_ms": latency_ms,
                "ruling": ruling,
            }
        )
        if options is not UNSET:
            field_dict["options"] = options
        if paused_ms is not UNSET:
            field_dict["paused_ms"] = paused_ms
        if point is not UNSET:
            field_dict["point"] = point

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.floor_option import FloorOption
        from ..models.floor_ruling import FloorRuling

        d = dict(src_dict)
        acted = d.pop("acted")

        at = datetime.datetime.fromisoformat(d.pop("at"))

        kind = FloorCheckKind._decode(d.pop("kind"))

        latency_ms = d.pop("latency_ms")

        ruling = FloorRuling.from_dict(d.pop("ruling"))

        _options = d.pop("options", UNSET)
        options: list[FloorOption] | Unset = UNSET
        if _options is not UNSET:
            options = []
            for options_item_data in _options:
                options_item = FloorOption.from_dict(options_item_data)

                options.append(options_item)

        paused_ms = d.pop("paused_ms", UNSET)

        point = d.pop("point", UNSET)

        floor_check = cls(
            acted=acted,
            at=at,
            kind=kind,
            latency_ms=latency_ms,
            ruling=ruling,
            options=options,
            paused_ms=paused_ms,
            point=point,
        )

        floor_check.additional_properties = d
        return floor_check

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
