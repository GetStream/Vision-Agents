from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.channel_kind import ChannelKind
from ..types import UNSET, Unset

T = TypeVar("T", bound="ChannelAccount")


@_attrs_define
class ChannelAccount:
    """One line the app answers on: the number people write to and where its provider delivers. The credentials are write-
    only, so they are never shown here.

        Attributes:
            created_at (datetime.datetime):
            delivering (bool): True when the provider was pointed at the webhook URL for you, so there is nothing left to
                paste.
            id (str):
            kind (ChannelKind): A channel a conversation can be carried over.
            number (str): The number people write to, in E.164.
            updated_at (datetime.datetime):
            webhook_url (str): Where the provider should deliver. Paste it into Meta's or Linq's webhook setup; a Telnyx
                number is pointed at it for you.
            account_id (str | Unset): The provider's own id for the line, such as WhatsApp's phone number id.
    """

    created_at: datetime.datetime
    delivering: bool
    id: str
    kind: ChannelKind
    number: str
    updated_at: datetime.datetime
    webhook_url: str
    account_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        delivering = self.delivering

        id = self.id

        kind = self.kind.value

        number = self.number

        updated_at = self.updated_at.isoformat()

        webhook_url = self.webhook_url

        account_id = self.account_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "delivering": delivering,
                "id": id,
                "kind": kind,
                "number": number,
                "updated_at": updated_at,
                "webhook_url": webhook_url,
            }
        )
        if account_id is not UNSET:
            field_dict["account_id"] = account_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        delivering = d.pop("delivering")

        id = d.pop("id")

        kind = ChannelKind(d.pop("kind"))

        number = d.pop("number")

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        webhook_url = d.pop("webhook_url")

        account_id = d.pop("account_id", UNSET)

        channel_account = cls(
            created_at=created_at,
            delivering=delivering,
            id=id,
            kind=kind,
            number=number,
            updated_at=updated_at,
            webhook_url=webhook_url,
            account_id=account_id,
        )

        channel_account.additional_properties = d
        return channel_account

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
