from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

if TYPE_CHECKING:
    from ..models.call_usage import CallUsage


T = TypeVar("T", bound="Run")


@_attrs_define
class Run:
    """
    Attributes:
        final_text (str): The last thing the agent said.
        model_route (str): The provider/model routing picked to answer.
        session_id (str): The session the run held. It ended with the run, and being incognito, it cannot be read back.
        usage (CallUsage): What the call spent, summed over every request it made. Counted once the call is over, so it
            is absent while one is still running. Requests that failed are included: a model that read the prompt and then
            fell over is still billed for it.
    """

    final_text: str
    model_route: str
    session_id: str
    usage: CallUsage
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        final_text = self.final_text

        model_route = self.model_route

        session_id = self.session_id

        usage = self.usage.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "final_text": final_text,
                "model_route": model_route,
                "session_id": session_id,
                "usage": usage,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.call_usage import CallUsage

        d = dict(src_dict)
        final_text = d.pop("final_text")

        model_route = d.pop("model_route")

        session_id = d.pop("session_id")

        usage = CallUsage.from_dict(d.pop("usage"))

        run = cls(
            final_text=final_text,
            model_route=model_route,
            session_id=session_id,
            usage=usage,
        )

        run.additional_properties = d
        return run

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
