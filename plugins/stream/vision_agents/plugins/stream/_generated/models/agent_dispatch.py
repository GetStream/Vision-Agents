from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.dispatch_setting import DispatchSetting
from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentDispatch")


@_attrs_define
class AgentDispatch:
    """What the agent leaves to the customer's own server, which waits on /v1/dispatch. Omitted settings are disabled.

    Attributes:
        incoming_call (DispatchSetting | Unset): Whether this kind of work is left to the customer's own dispatch
            worker.
        text (DispatchSetting | Unset): Whether this kind of work is left to the customer's own dispatch worker.
    """

    incoming_call: DispatchSetting | Unset = UNSET
    text: DispatchSetting | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        incoming_call: str | Unset = UNSET
        if not isinstance(self.incoming_call, Unset):
            incoming_call = self.incoming_call.value

        text: str | Unset = UNSET
        if not isinstance(self.text, Unset):
            text = self.text.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if incoming_call is not UNSET:
            field_dict["incoming_call"] = incoming_call
        if text is not UNSET:
            field_dict["text"] = text

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        _incoming_call = d.pop("incoming_call", UNSET)
        incoming_call: DispatchSetting | Unset
        if isinstance(_incoming_call, Unset):
            incoming_call = UNSET
        else:
            incoming_call = DispatchSetting(_incoming_call)

        _text = d.pop("text", UNSET)
        text: DispatchSetting | Unset
        if isinstance(_text, Unset):
            text = UNSET
        else:
            text = DispatchSetting(_text)

        agent_dispatch = cls(
            incoming_call=incoming_call,
            text=text,
        )

        agent_dispatch.additional_properties = d
        return agent_dispatch

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
