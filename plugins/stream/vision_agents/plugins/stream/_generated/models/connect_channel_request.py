from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectChannelRequest")


@_attrs_define
class ConnectChannelRequest:
    """Credentials for one line, connected once for the app and named by any number of agents under channels in agent.yaml.
    Sending a line already connected replaces its credentials and keeps its webhook URL.

        Attributes:
            kind (str): whatsapp, sms or imessage.
            number (str): The number people write to, in E.164. For sms it must be a number this app bought with POST
                /v1/phone/numbers.
            account_id (str | Unset): The provider's own id for the line. WhatsApp's phone number id; not needed by the
                others.
            challenge (str | Unset): The verify token Meta's webhook setup echoes back. WhatsApp only.
            signing (str | Unset): What the provider signs deliveries with: Meta's app secret, Telnyx's public key, Linq's
                whsec_ secret.
            token (str | Unset): What authenticates a send: a Meta access token, a Telnyx API key, a Linq API key.
    """

    kind: str
    number: str
    account_id: str | Unset = UNSET
    challenge: str | Unset = UNSET
    signing: str | Unset = UNSET
    token: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind

        number = self.number

        account_id = self.account_id

        challenge = self.challenge

        signing = self.signing

        token = self.token

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "kind": kind,
                "number": number,
            }
        )
        if account_id is not UNSET:
            field_dict["account_id"] = account_id
        if challenge is not UNSET:
            field_dict["challenge"] = challenge
        if signing is not UNSET:
            field_dict["signing"] = signing
        if token is not UNSET:
            field_dict["token"] = token

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        kind = d.pop("kind")

        number = d.pop("number")

        account_id = d.pop("account_id", UNSET)

        challenge = d.pop("challenge", UNSET)

        signing = d.pop("signing", UNSET)

        token = d.pop("token", UNSET)

        connect_channel_request = cls(
            kind=kind,
            number=number,
            account_id=account_id,
            challenge=challenge,
            signing=signing,
            token=token,
        )

        connect_channel_request.additional_properties = d
        return connect_channel_request

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
