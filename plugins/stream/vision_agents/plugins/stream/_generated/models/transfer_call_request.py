from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.transfer_call_request_tags import TransferCallRequestTags


T = TypeVar("T", bound="TransferCallRequest")


@_attrs_define
class TransferCallRequest:
    """
    Attributes:
        from_ (str): The customer's number the human is dialled from, which is what they see.
        to (str): The human being brought onto the call.
        call_id (str): The Stream call the caller and the agent are already on.
        call_type (str | Unset): The Stream call type. Omit for "default".
        tags (TransferCallRequestTags | Unset):
    """

    from_: str
    to: str
    call_id: str
    call_type: str | Unset = UNSET
    tags: TransferCallRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_ = self.from_

        to = self.to

        call_id = self.call_id

        call_type = self.call_type

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "from": from_,
                "to": to,
                "call_id": call_id,
            }
        )
        if call_type is not UNSET:
            field_dict["call_type"] = call_type
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.transfer_call_request_tags import TransferCallRequestTags

        d = dict(src_dict)
        from_ = d.pop("from")

        to = d.pop("to")

        call_id = d.pop("call_id")

        call_type = d.pop("call_type", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: TransferCallRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = TransferCallRequestTags.from_dict(_tags)

        transfer_call_request = cls(
            from_=from_,
            to=to,
            call_id=call_id,
            call_type=call_type,
            tags=tags,
        )

        transfer_call_request.additional_properties = d
        return transfer_call_request

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
