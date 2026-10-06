from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.stream_key_state_signs_webhooks import StreamKeyStateSignsWebhooks
from ..models.stream_key_state_status import StreamKeyStateStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="StreamKeyState")


@_attrs_define
class StreamKeyState:
    """
    Attributes:
        api_key (str):
        signs_webhooks (StreamKeyStateSignsWebhooks): Whether Stream signs the app's hooks with this key, which is its
            oldest. yes is one a hook arrived signed with.
        status (StreamKeyStateStatus): rejected is a key Stream stopped accepting, which the router no longer uses.
        created_at (datetime.datetime | Unset): When Stream made the key.
        last_webhook_at (datetime.datetime | Unset): When Stream last signed a hook with this key.
        secret_last4 (str | Unset): The end of the secret, enough to tell two apart.
        verified_at (datetime.datetime | Unset):
    """

    api_key: str
    signs_webhooks: StreamKeyStateSignsWebhooks
    status: StreamKeyStateStatus
    created_at: datetime.datetime | Unset = UNSET
    last_webhook_at: datetime.datetime | Unset = UNSET
    secret_last4: str | Unset = UNSET
    verified_at: datetime.datetime | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        api_key = self.api_key

        signs_webhooks = self.signs_webhooks.value

        status = self.status.value

        created_at: str | Unset = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.isoformat()

        last_webhook_at: str | Unset = UNSET
        if not isinstance(self.last_webhook_at, Unset):
            last_webhook_at = self.last_webhook_at.isoformat()

        secret_last4 = self.secret_last4

        verified_at: str | Unset = UNSET
        if not isinstance(self.verified_at, Unset):
            verified_at = self.verified_at.isoformat()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "api_key": api_key,
                "signs_webhooks": signs_webhooks,
                "status": status,
            }
        )
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if last_webhook_at is not UNSET:
            field_dict["last_webhook_at"] = last_webhook_at
        if secret_last4 is not UNSET:
            field_dict["secret_last4"] = secret_last4
        if verified_at is not UNSET:
            field_dict["verified_at"] = verified_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        api_key = d.pop("api_key")

        signs_webhooks = StreamKeyStateSignsWebhooks(d.pop("signs_webhooks"))

        status = StreamKeyStateStatus(d.pop("status"))

        _created_at = d.pop("created_at", UNSET)
        created_at: datetime.datetime | Unset
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = datetime.datetime.fromisoformat(_created_at)

        _last_webhook_at = d.pop("last_webhook_at", UNSET)
        last_webhook_at: datetime.datetime | Unset
        if isinstance(_last_webhook_at, Unset):
            last_webhook_at = UNSET
        else:
            last_webhook_at = datetime.datetime.fromisoformat(_last_webhook_at)

        secret_last4 = d.pop("secret_last4", UNSET)

        _verified_at = d.pop("verified_at", UNSET)
        verified_at: datetime.datetime | Unset
        if isinstance(_verified_at, Unset):
            verified_at = UNSET
        else:
            verified_at = datetime.datetime.fromisoformat(_verified_at)

        stream_key_state = cls(
            api_key=api_key,
            signs_webhooks=signs_webhooks,
            status=status,
            created_at=created_at,
            last_webhook_at=last_webhook_at,
            secret_last4=secret_last4,
            verified_at=verified_at,
        )

        stream_key_state.additional_properties = d
        return stream_key_state

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
