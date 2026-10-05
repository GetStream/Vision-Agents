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
        attempts (int):
        id (str):
        state (ContactState):
        to_number (str):
        call_id (str | Unset): The call this contact became, which is what the call paths take.
        error (str | Unset): Why they could not be rung, when they could not be.
        instructions (str | Unset):
        vendor_call_id (str | Unset):
    """

    attempts: int
    id: str
    state: ContactState
    to_number: str
    call_id: str | Unset = UNSET
    error: str | Unset = UNSET
    instructions: str | Unset = UNSET
    vendor_call_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        attempts = self.attempts

        id = self.id

        state = self.state.value

        to_number = self.to_number

        call_id = self.call_id

        error = self.error

        instructions = self.instructions

        vendor_call_id = self.vendor_call_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "attempts": attempts,
                "id": id,
                "state": state,
                "to_number": to_number,
            }
        )
        if call_id is not UNSET:
            field_dict["call_id"] = call_id
        if error is not UNSET:
            field_dict["error"] = error
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if vendor_call_id is not UNSET:
            field_dict["vendor_call_id"] = vendor_call_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        attempts = d.pop("attempts")

        id = d.pop("id")

        state = ContactState(d.pop("state"))

        to_number = d.pop("to_number")

        call_id = d.pop("call_id", UNSET)

        error = d.pop("error", UNSET)

        instructions = d.pop("instructions", UNSET)

        vendor_call_id = d.pop("vendor_call_id", UNSET)

        contact = cls(
            attempts=attempts,
            id=id,
            state=state,
            to_number=to_number,
            call_id=call_id,
            error=error,
            instructions=instructions,
            vendor_call_id=vendor_call_id,
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
