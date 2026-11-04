from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.data_policy import DataPolicy


T = TypeVar("T", bound="OcrOptions")


@_attrs_define
class OcrOptions:
    """Who reads the page, and what may happen to it afterwards.

    Attributes:
        data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
            transcribed, or the text they had spoken and the voice speaking it, and the region it is done in. This is a
            requirement rather than a description: a request naming one is only routed to a model whose declared handling
            meets it, and if none does the request is refused rather than sent somewhere that does not.
        providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
            it holds anything.
        target (str | Unset): A provider/model or a shortcut. Defaults to ocr-default. Example: ocr-vision.
    """

    data_policy: DataPolicy | Unset = UNSET
    providers: list[str] | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if providers is not UNSET:
            field_dict["providers"] = providers
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.data_policy import DataPolicy

        d = dict(src_dict)
        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        providers = cast(list[str], d.pop("providers", UNSET))

        target = d.pop("target", UNSET)

        ocr_options = cls(
            data_policy=data_policy,
            providers=providers,
            target=target,
        )

        ocr_options.additional_properties = d
        return ocr_options

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
