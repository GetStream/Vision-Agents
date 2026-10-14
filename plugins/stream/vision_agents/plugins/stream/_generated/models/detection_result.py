from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.detection_error_code import DetectionErrorCode
from ..models.detection_status import DetectionStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.detected_object import DetectedObject


T = TypeVar("T", bound="DetectionResult")


@_attrs_define
class DetectionResult:
    """
    Attributes:
        cost_micros (int): Millionths of a dollar, priced per image the vendor billed. Zero when it failed.
        detections (list[DetectedObject]): What was found. Empty when nothing was, or when it failed.
        id (str): This response's own id, for logs. Nothing is stored under it. Example: det_1b9d6bcd-
            bbfd-4b2d-9b5d-ab8dfbbd4bed.
        status (DetectionStatus): Whether the image was looked at. A failed detection says why in error_code and error.
        error (str | Unset): What went wrong, in words. Absent when the detection completed.
        error_code (DetectionErrorCode | Unset): Why a detection found nothing, absent when it completed.
            unsupported_option is a class, setting or overwrite no candidate could honour; provider_failed anything else a
            provider did wrong; timeout the 30 seconds running out; cancelled the caller hanging up.
        model (str | Unset):  Example: moondream3-preview.
        provider (str | Unset): Who looked, or who failed to. Absent when nothing got as far as a provider. Example:
            moondream.
    """

    cost_micros: int
    detections: list[DetectedObject]
    id: str
    status: DetectionStatus
    error: str | Unset = UNSET
    error_code: DetectionErrorCode | Unset = UNSET
    model: str | Unset = UNSET
    provider: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_micros = self.cost_micros

        detections = []
        for detections_item_data in self.detections:
            detections_item = detections_item_data.to_dict()
            detections.append(detections_item)

        id = self.id

        status = self.status.value

        error = self.error

        error_code: str | Unset = UNSET
        if not isinstance(self.error_code, Unset):
            error_code = self.error_code.value

        model = self.model

        provider = self.provider

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cost_micros": cost_micros,
                "detections": detections,
                "id": id,
                "status": status,
            }
        )
        if error is not UNSET:
            field_dict["error"] = error
        if error_code is not UNSET:
            field_dict["error_code"] = error_code
        if model is not UNSET:
            field_dict["model"] = model
        if provider is not UNSET:
            field_dict["provider"] = provider

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.detected_object import DetectedObject

        d = dict(src_dict)
        cost_micros = d.pop("cost_micros")

        detections = []
        _detections = d.pop("detections")
        for detections_item_data in _detections:
            detections_item = DetectedObject.from_dict(detections_item_data)

            detections.append(detections_item)

        id = d.pop("id")

        status = DetectionStatus(d.pop("status"))

        error = d.pop("error", UNSET)

        _error_code = d.pop("error_code", UNSET)
        error_code: DetectionErrorCode | Unset
        if isinstance(_error_code, Unset):
            error_code = UNSET
        else:
            error_code = DetectionErrorCode(_error_code)

        model = d.pop("model", UNSET)

        provider = d.pop("provider", UNSET)

        detection_result = cls(
            cost_micros=cost_micros,
            detections=detections,
            id=id,
            status=status,
            error=error,
            error_code=error_code,
            model=model,
            provider=provider,
        )

        detection_result.additional_properties = d
        return detection_result

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
