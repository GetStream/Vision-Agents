from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentUser")


@_attrs_define
class AgentUser:
    """
    Attributes:
        phone (None | str): The number linked to the user, in E.164; null when none is. Example: +15551234567.
        user_id (str): The Stream user id, as the app's own token signs it.
        include_phone_history (bool | Unset): true when the user reads all of what the number ever said. Left out
            otherwise.
        phone_history_since (datetime.datetime | Unset): From when on the user reads what the number said: the link's
            time, the time of a request with include_phone_history false, or the phone_history_since sent. Left out with
            include_phone_history true, or with no number.
    """

    phone: None | str
    user_id: str
    include_phone_history: bool | Unset = UNSET
    phone_history_since: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        phone: None | str
        phone = self.phone

        user_id = self.user_id

        include_phone_history = self.include_phone_history

        phone_history_since: str | Unset = UNSET
        if not isinstance(self.phone_history_since, Unset):
            phone_history_since = self.phone_history_since.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "phone": phone,
                "user_id": user_id,
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

        user_id = d.pop("user_id")

        include_phone_history = d.pop("include_phone_history", UNSET)

        _phone_history_since = d.pop("phone_history_since", UNSET)
        phone_history_since: datetime.datetime | Unset
        if isinstance(_phone_history_since, Unset):
            phone_history_since = UNSET
        else:
            phone_history_since = datetime.datetime.fromisoformat(_phone_history_since)

        agent_user = cls(
            phone=phone,
            user_id=user_id,
            include_phone_history=include_phone_history,
            phone_history_since=phone_history_since,
        )

        agent_user.additional_properties = d
        return agent_user

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
