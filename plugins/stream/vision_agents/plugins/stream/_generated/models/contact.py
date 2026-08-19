from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.contact_state import ContactState
from ..types import UNSET, Unset

T = TypeVar("T", bound="Contact")


@_attrs_define
class Contact:
    """
    Attributes:
        id (str):
        to_number (str):
        state (ContactState):
        attempts (int):
        instructions (str | Unset):
        call_id (str | Unset): The call this contact became, which is what the call paths take.
        vendor_call_id (str | Unset):
        error (str | Unset): Why they could not be rung, when they could not be.
    """

    id: str
    to_number: str
    state: ContactState
    attempts: int
    instructions: str | Unset = UNSET
    call_id: str | Unset = UNSET
    vendor_call_id: str | Unset = UNSET
    error: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        to_number = self.to_number

        state = self.state.value

        attempts = self.attempts

        instructions = self.instructions

        call_id = self.call_id

        vendor_call_id = self.vendor_call_id

        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "to_number": to_number,
                "state": state,
                "attempts": attempts,
            }
        )
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if vendor_call_id is not UNSET:
            field_dict["vendor_call_id"] = vendor_call_id
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        to_number = d.pop("to_number")

        state = ContactState(d.pop("state"))

        attempts = d.pop("attempts")

        instructions = d.pop("instructions", UNSET)

        call_id = d.pop("call_id", UNSET)

        vendor_call_id = d.pop("vendor_call_id", UNSET)

        error = d.pop("error", UNSET)

        contact = cls(
            id=id,
            to_number=to_number,
            state=state,
            attempts=attempts,
            instructions=instructions,
            call_id=call_id,
            vendor_call_id=vendor_call_id,
            error=error,
        )

        contact.additional_properties = d
        return contact

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
