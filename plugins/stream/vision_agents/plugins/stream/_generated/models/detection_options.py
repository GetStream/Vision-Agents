from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.detection_options_overwrites import DetectionOptionsOverwrites


T = TypeVar("T", bound="DetectionOptions")


@_attrs_define
class DetectionOptions:
    """Where to look and what to keep.

    Attributes:
        classes (list[str] | Unset): The labels to keep. An open-vocabulary model (detection-open) looks for these words
            and has to be given at least one; a fixed-class model returns only these of what it found. Matched without
            regard to case.
        embeddings (bool | Unset): Return a vector on every box, for matching what was found against what was seen
            before. Only a model that computes one is asked.
        min_confidence (float | Unset): Drop boxes scored below this. Only a model that scores its boxes is asked.
        overwrites (DetectionOptionsOverwrites | Unset): Settings for one provider that the options above have no word
            for, keyed by provider. roboflow takes workspace and workflow_id, or specification, for the roboflow/workflow
            model. An overwrite for a provider that is not a candidate is refused.
        providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
            it holds anything.
        target (str | Unset): A provider/model or a capability shortcut. Defaults to detection-fast. Example: detection-
            open.
    """

    classes: list[str] | Unset = UNSET
    embeddings: bool | Unset = UNSET
    min_confidence: float | Unset = UNSET
    overwrites: DetectionOptionsOverwrites | Unset = UNSET
    providers: list[str] | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        classes: list[str] | Unset = UNSET
        if not isinstance(self.classes, Unset):
            classes = self.classes

        embeddings = self.embeddings

        min_confidence = self.min_confidence

        overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.overwrites, Unset):
            overwrites = self.overwrites.to_dict()

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if classes is not UNSET:
            field_dict["classes"] = classes
        if embeddings is not UNSET:
            field_dict["embeddings"] = embeddings
        if min_confidence is not UNSET:
            field_dict["min_confidence"] = min_confidence
        if overwrites is not UNSET:
            field_dict["overwrites"] = overwrites
        if providers is not UNSET:
            field_dict["providers"] = providers
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.detection_options_overwrites import DetectionOptionsOverwrites

        d = dict(src_dict)
        classes = cast(list[str], d.pop("classes", UNSET))

        embeddings = d.pop("embeddings", UNSET)

        min_confidence = d.pop("min_confidence", UNSET)

        _overwrites = d.pop("overwrites", UNSET)
        overwrites: DetectionOptionsOverwrites | Unset
        if isinstance(_overwrites, Unset):
            overwrites = UNSET
        else:
            overwrites = DetectionOptionsOverwrites.from_dict(_overwrites)

        providers = cast(list[str], d.pop("providers", UNSET))

        target = d.pop("target", UNSET)

        detection_options = cls(
            classes=classes,
            embeddings=embeddings,
            min_confidence=min_confidence,
            overwrites=overwrites,
            providers=providers,
            target=target,
        )

        detection_options.additional_properties = d
        return detection_options

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
