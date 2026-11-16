from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.end_call_settings import EndCallSettings
    from ..models.press_settings import PressSettings
    from ..models.transfer_settings import TransferSettings


T = TypeVar("T", bound="AgentCalls")


@_attrs_define
class AgentCalls:
    """The tools an agent on a phone call is offered: end the call, press keys, transfer.

    Attributes:
        end_call (EndCallSettings | Unset):
        press (PressSettings | Unset):
        transfer (TransferSettings | Unset):
    """

    end_call: EndCallSettings | Unset = UNSET
    press: PressSettings | Unset = UNSET
    transfer: TransferSettings | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        end_call: dict[str, Any] | Unset = UNSET
        if not isinstance(self.end_call, Unset):
            end_call = self.end_call.to_dict()

        press: dict[str, Any] | Unset = UNSET
        if not isinstance(self.press, Unset):
            press = self.press.to_dict()

        transfer: dict[str, Any] | Unset = UNSET
        if not isinstance(self.transfer, Unset):
            transfer = self.transfer.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if end_call is not UNSET:
            field_dict["end_call"] = end_call
        if press is not UNSET:
            field_dict["press"] = press
        if transfer is not UNSET:
            field_dict["transfer"] = transfer

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.end_call_settings import EndCallSettings
        from ..models.press_settings import PressSettings
        from ..models.transfer_settings import TransferSettings

        d = dict(src_dict)
        _end_call = d.pop("end_call", UNSET)
        end_call: EndCallSettings | Unset
        if isinstance(_end_call, Unset):
            end_call = UNSET
        else:
            end_call = EndCallSettings.from_dict(_end_call)

        _press = d.pop("press", UNSET)
        press: PressSettings | Unset
        if isinstance(_press, Unset):
            press = UNSET
        else:
            press = PressSettings.from_dict(_press)

        _transfer = d.pop("transfer", UNSET)
        transfer: TransferSettings | Unset
        if isinstance(_transfer, Unset):
            transfer = UNSET
        else:
            transfer = TransferSettings.from_dict(_transfer)

        agent_calls = cls(
            end_call=end_call,
            press=press,
            transfer=transfer,
        )

        agent_calls.additional_properties = d
        return agent_calls

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
