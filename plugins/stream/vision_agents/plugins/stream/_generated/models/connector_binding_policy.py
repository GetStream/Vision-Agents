from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.connector_on_interrupt import ConnectorOnInterrupt
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectorBindingPolicy")


@_attrs_define
class ConnectorBindingPolicy:
    """How a binding's tool calls behave around speech and interruptions. Every field is optional, and a field left out
    keeps today's behaviour.

        Attributes:
            cancellable (bool | Unset): Whether the provider is told to stop a call the session stopped waiting for. Omitted
                is true. False leaves it running after an interruption, for a tool that is not safe to stop halfway, such as a
                payment; the binding's timeout still ends it and tells the provider to stop it. It only matters with
                on_interrupt cancel: a wait call is never stopped by an interruption.
            on_interrupt (ConnectorOnInterrupt | Unset): cancel stops waiting for the call when the turn is interrupted, and
                tells the provider to stop it unless cancellable is false. wait lets the call finish, up to the binding's
                timeout, and its result goes into the conversation for the next turn.
            pre_speech (str | Unset): What the agent says while one of the binding's tools runs, such as "Let me pull up
                your calendar.", in place of the phrase it picks itself when the model reached for the tool without a word. A
                voice session with a separate voice says it; every session reports it on tool_started.
    """

    cancellable: bool | Unset = UNSET
    on_interrupt: ConnectorOnInterrupt | Unset = UNSET
    pre_speech: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cancellable = self.cancellable

        on_interrupt: str | Unset = UNSET
        if not isinstance(self.on_interrupt, Unset):
            on_interrupt = self.on_interrupt.value

        pre_speech = self.pre_speech

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if cancellable is not UNSET:
            field_dict["cancellable"] = cancellable
        if on_interrupt is not UNSET:
            field_dict["on_interrupt"] = on_interrupt
        if pre_speech is not UNSET:
            field_dict["pre_speech"] = pre_speech

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        cancellable = d.pop("cancellable", UNSET)

        _on_interrupt = d.pop("on_interrupt", UNSET)
        on_interrupt: ConnectorOnInterrupt | Unset
        if isinstance(_on_interrupt, Unset):
            on_interrupt = UNSET
        else:
            on_interrupt = ConnectorOnInterrupt(_on_interrupt)

        pre_speech = d.pop("pre_speech", UNSET)

        connector_binding_policy = cls(
            cancellable=cancellable,
            on_interrupt=on_interrupt,
            pre_speech=pre_speech,
        )

        connector_binding_policy.additional_properties = d
        return connector_binding_policy

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
