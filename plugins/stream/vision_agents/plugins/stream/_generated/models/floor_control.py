from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="FloorControl")


@_attrs_define
class FloorControl:
    """The model that controls an agent's floor, and how long a caller may wait once they stop. Left out or null, the
    stored choice stays; an empty object is no choice again, which the router's default rules: the flow controller, or
    stream/chorus-asr at the server's default point where the router defaults every agent to it; {model: flow-
    controller} keeps the flow controller whatever the router's default; anything else replaces the stored choice whole.

        Attributes:
            max_wait_ms (int | Unset): The most a caller who has finished may wait for the answer, in milliseconds. It
                chooses the point of the model's measured trade-off with the fewest cut-ins whose measured p90 wait is within
                it, and is refused (400) when it is shorter than every point's. Left out, the server's default point. It is
                stream/chorus-asr's: with flow-controller it is refused (400). Example: 800.
            model (str | Unset): The model that controls the floor, hearing the caller's audio: when they have finished, and
                what to do about speech over the agent. stream/chorus-asr is the one model there is; flow-controller keeps the
                flow controller, which rules the floor where no model does, whatever the router defaults an agent that chose
                none to. Required, unless floor_control is empty. stream/chorus-asr on a router without a chorus-asr server is
                refused (400 not_configured). Example: stream/chorus-asr.
    """

    max_wait_ms: int | Unset = UNSET
    model: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        max_wait_ms = self.max_wait_ms

        model = self.model

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if max_wait_ms is not UNSET:
            field_dict["max_wait_ms"] = max_wait_ms
        if model is not UNSET:
            field_dict["model"] = model

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        max_wait_ms = d.pop("max_wait_ms", UNSET)

        model = d.pop("model", UNSET)

        floor_control = cls(
            max_wait_ms=max_wait_ms,
            model=model,
        )

        return floor_control
