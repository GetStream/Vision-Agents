from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

T = TypeVar("T", bound="AgentUserRequest")


@_attrs_define
class AgentUserRequest:
    """Who one of the app's users is, as the app's backend knows it. phone is required: a number, or null for none.

    Attributes:
        phone (None | str): The user's phone number in E.164, with its plus. Their SMS, WhatsApp messages, iMessages and
            calls from it then share one omni-channel with their app chat, under each agent of the app: the agent's episode
            cards of one are read by the other. null unlinks the number the user had; what was joined before stays joined. A
            new number takes the place of the old one. A number linked to another user of the app is refused (409
            phone_taken). Example: +15551234567.
    """

    phone: None | str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        phone: None | str
        phone = self.phone

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "phone": phone,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)

        def _parse_phone(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        phone = _parse_phone(d.pop("phone"))

        agent_user_request = cls(
            phone=phone,
        )

        agent_user_request.additional_properties = d
        return agent_user_request

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
