from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.floor_control_mode import FloorControlMode
from ..models.floor_control_model import FloorControlModel

T = TypeVar("T", bound="FloorControl")


@_attrs_define
class FloorControl:
    """Whether a model that hears the caller's audio rules the agent's floor: when a caller has finished, and what to do
    about speech over the agent. An agent opts in for itself, and one that never chose is off. shadow and act need the
    router's chorus-asr server, and a write asking for either on a router without one is refused. Off, the flow
    controller rules the floor as it does with no such model, and end_of_turn does nothing; chorus-asr still transcribes
    for an agent whose stt names stream/chorus-asr. Each session runs by its own, so agents on one router may choose
    differently.

        Attributes:
            mode (FloorControlMode): What the agent does with the model's rulings. off asks it nothing. shadow asks it about
                each turn beside the flow controller and acts on nothing it says, to compare the two on real calls. act asks it
                first and acts on its ruling, and the flow controller rules a turn it does not answer in time.
            model (FloorControlModel): A model that hears the caller's audio and rules an agent's floor. chorus-asr is the
                one there is.
    """

    mode: FloorControlMode
    model: FloorControlModel

    def to_dict(self) -> dict[str, Any]:
        mode = self.mode.value

        model = self.model.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "mode": mode,
                "model": model,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        mode = FloorControlMode(d.pop("mode"))

        model = FloorControlModel(d.pop("model"))

        floor_control = cls(
            mode=mode,
            model=model,
        )

        return floor_control
