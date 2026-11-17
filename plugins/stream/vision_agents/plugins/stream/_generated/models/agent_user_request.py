from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentUserRequest")


@_attrs_define
class AgentUserRequest:
    """Who one of the app's users is, as the app's backend knows it. phone is required: a number, or null for none.

    Attributes:
        phone (None | str): The user's phone number in E.164, with its plus. From then on, their SMS, WhatsApp messages,
            iMessages and calls from it share one omni-channel with their app chat, under each agent of the app: the agent's
            episode cards of one are read by the other. What the number said before the link is not the user's, unless
            include_phone_history or phone_history_since says otherwise: the user reads none of its earlier cards, whether
            the agent had episode_cards on at the link or turns them on later, and a session on the number's SMS, WhatsApp
            or iMessage thread is handed none of the thread's earlier messages. null unlinks the number the user had, and
            that number then reads none of the user's cards. A new number takes the place of the old one, which reads none
            of the user's cards from then on. A number linked to another user of the app is refused (409 phone_taken).
            Example: +15551234567.
        include_phone_history (bool | Unset): How much of what the number said before this request the user reads. true:
            all of it, ever. false: none of it, from this request on. Left out with phone_history_since left out too: from
            the link on for a number new to the user, and what was stored for the number the user has. true with
            phone_history_since is refused (400); false with it, phone_history_since applies.
        phone_history_since (datetime.datetime | Unset): The user reads what the number said from this time on, in RFC
            3339, such as 2026-11-02T00:00:00Z: a conversation that started before it is left out whole. A time in the
            future is refused (400).
    """

    phone: None | str
    include_phone_history: bool | Unset = UNSET
    phone_history_since: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        phone: None | str
        phone = self.phone

        include_phone_history = self.include_phone_history

        phone_history_since: str | Unset = UNSET
        if not isinstance(self.phone_history_since, Unset):
            phone_history_since = self.phone_history_since.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "phone": phone,
            }
        )
        if include_phone_history is not UNSET:
            field_dict["include_phone_history"] = include_phone_history
        if phone_history_since is not UNSET:
            field_dict["phone_history_since"] = phone_history_since

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)

        def _parse_phone(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        phone = _parse_phone(d.pop("phone"))

        include_phone_history = d.pop("include_phone_history", UNSET)

        _phone_history_since = d.pop("phone_history_since", UNSET)
        phone_history_since: datetime.datetime | Unset
        if isinstance(_phone_history_since, Unset):
            phone_history_since = UNSET
        else:
            phone_history_since = datetime.datetime.fromisoformat(_phone_history_since)

        agent_user_request = cls(
            phone=phone,
            include_phone_history=include_phone_history,
            phone_history_since=phone_history_since,
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
