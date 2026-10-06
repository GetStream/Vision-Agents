from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.channel_identity import ChannelIdentity
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.channel_line_request import ChannelLineRequest


T = TypeVar("T", bound="AgentChannels")


@_attrs_define
class AgentChannels:
    """The lines this agent answers on besides its Stream Chat channel. Each names a number the app connected with POST
    /v1/agents/channels, and only one agent may answer on a number. A message that arrives is answered in the sender's
    own conversation, so what they say is kept and shown wherever the rest of it is.

        Attributes:
            identity (ChannelIdentity | Unset): How a sender becomes an end user. phone makes each number an end user of its
                own, phone:+15551234567, so anybody who writes is answered. link answers only a number somebody tied to an end
                user with a code from POST /v1/agents/channels/links, which is what an agent reading a person's own calendar or
                orders needs. Omitted is phone.
            imessage (ChannelLineRequest | Unset):
            sms (ChannelLineRequest | Unset):
            whatsapp (ChannelLineRequest | Unset):
    """

    identity: ChannelIdentity | Unset = UNSET
    imessage: ChannelLineRequest | Unset = UNSET
    sms: ChannelLineRequest | Unset = UNSET
    whatsapp: ChannelLineRequest | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        identity: str | Unset = UNSET
        if not isinstance(self.identity, Unset):
            identity = self.identity.value

        imessage: dict[str, Any] | Unset = UNSET
        if not isinstance(self.imessage, Unset):
            imessage = self.imessage.to_dict()

        sms: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sms, Unset):
            sms = self.sms.to_dict()

        whatsapp: dict[str, Any] | Unset = UNSET
        if not isinstance(self.whatsapp, Unset):
            whatsapp = self.whatsapp.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if identity is not UNSET:
            field_dict["identity"] = identity
        if imessage is not UNSET:
            field_dict["imessage"] = imessage
        if sms is not UNSET:
            field_dict["sms"] = sms
        if whatsapp is not UNSET:
            field_dict["whatsapp"] = whatsapp

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.channel_line_request import ChannelLineRequest

        d = dict(src_dict)
        _identity = d.pop("identity", UNSET)
        identity: ChannelIdentity | Unset
        if isinstance(_identity, Unset):
            identity = UNSET
        else:
            identity = ChannelIdentity(_identity)

        _imessage = d.pop("imessage", UNSET)
        imessage: ChannelLineRequest | Unset
        if isinstance(_imessage, Unset):
            imessage = UNSET
        else:
            imessage = ChannelLineRequest.from_dict(_imessage)

        _sms = d.pop("sms", UNSET)
        sms: ChannelLineRequest | Unset
        if isinstance(_sms, Unset):
            sms = UNSET
        else:
            sms = ChannelLineRequest.from_dict(_sms)

        _whatsapp = d.pop("whatsapp", UNSET)
        whatsapp: ChannelLineRequest | Unset
        if isinstance(_whatsapp, Unset):
            whatsapp = UNSET
        else:
            whatsapp = ChannelLineRequest.from_dict(_whatsapp)

        agent_channels = cls(
            identity=identity,
            imessage=imessage,
            sms=sms,
            whatsapp=whatsapp,
        )

        agent_channels.additional_properties = d
        return agent_channels

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
