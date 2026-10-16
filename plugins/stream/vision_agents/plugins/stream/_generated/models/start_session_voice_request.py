from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_call import SessionCall


T = TypeVar("T", bound="StartSessionVoiceRequest")


@_attrs_define
class StartSessionVoiceRequest:
    """
    Attributes:
        call (SessionCall | Unset): An existing Stream call for the agent to join, by its type and id.
    """

    call: SessionCall | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        call: dict[str, Any] | Unset = UNSET
        if not isinstance(self.call, Unset):
            call = self.call.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if call is not UNSET:
            field_dict["call"] = call

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_call import SessionCall

        d = dict(src_dict)
        _call = d.pop("call", UNSET)
        call: SessionCall | Unset
        if isinstance(_call, Unset):
            call = UNSET
        else:
            call = SessionCall.from_dict(_call)

        start_session_voice_request = cls(
            call=call,
        )

        start_session_voice_request.additional_properties = d
        return start_session_voice_request

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
