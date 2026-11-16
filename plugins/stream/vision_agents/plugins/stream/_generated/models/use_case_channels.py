from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.i_message_profile import IMessageProfile
    from ..models.rcs_profile import RCSProfile
    from ..models.voice_profile import VoiceProfile
    from ..models.whats_app_profile import WhatsAppProfile


T = TypeVar("T", bound="UseCaseChannels")


@_attrs_define
class UseCaseChannels:
    """
    Attributes:
        imessage (IMessageProfile | Unset):
        rcs (RCSProfile | Unset):
        voice (VoiceProfile | Unset):
        whatsapp (WhatsAppProfile | Unset):
    """

    imessage: IMessageProfile | Unset = UNSET
    rcs: RCSProfile | Unset = UNSET
    voice: VoiceProfile | Unset = UNSET
    whatsapp: WhatsAppProfile | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        imessage: dict[str, Any] | Unset = UNSET
        if not isinstance(self.imessage, Unset):
            imessage = self.imessage.to_dict()

        rcs: dict[str, Any] | Unset = UNSET
        if not isinstance(self.rcs, Unset):
            rcs = self.rcs.to_dict()

        voice: dict[str, Any] | Unset = UNSET
        if not isinstance(self.voice, Unset):
            voice = self.voice.to_dict()

        whatsapp: dict[str, Any] | Unset = UNSET
        if not isinstance(self.whatsapp, Unset):
            whatsapp = self.whatsapp.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if imessage is not UNSET:
            field_dict["imessage"] = imessage
        if rcs is not UNSET:
            field_dict["rcs"] = rcs
        if voice is not UNSET:
            field_dict["voice"] = voice
        if whatsapp is not UNSET:
            field_dict["whatsapp"] = whatsapp

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.i_message_profile import IMessageProfile
        from ..models.rcs_profile import RCSProfile
        from ..models.voice_profile import VoiceProfile
        from ..models.whats_app_profile import WhatsAppProfile

        d = dict(src_dict)
        _imessage = d.pop("imessage", UNSET)
        imessage: IMessageProfile | Unset
        if isinstance(_imessage, Unset):
            imessage = UNSET
        else:
            imessage = IMessageProfile.from_dict(_imessage)

        _rcs = d.pop("rcs", UNSET)
        rcs: RCSProfile | Unset
        if isinstance(_rcs, Unset):
            rcs = UNSET
        else:
            rcs = RCSProfile.from_dict(_rcs)

        _voice = d.pop("voice", UNSET)
        voice: VoiceProfile | Unset
        if isinstance(_voice, Unset):
            voice = UNSET
        else:
            voice = VoiceProfile.from_dict(_voice)

        _whatsapp = d.pop("whatsapp", UNSET)
        whatsapp: WhatsAppProfile | Unset
        if isinstance(_whatsapp, Unset):
            whatsapp = UNSET
        else:
            whatsapp = WhatsAppProfile.from_dict(_whatsapp)

        use_case_channels = cls(
            imessage=imessage,
            rcs=rcs,
            voice=voice,
            whatsapp=whatsapp,
        )

        use_case_channels.additional_properties = d
        return use_case_channels

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
