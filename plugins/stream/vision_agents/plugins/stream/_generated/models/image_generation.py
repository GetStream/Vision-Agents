from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.image_error_code import ImageErrorCode
from ..models.image_generation_status import ImageGenerationStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.generated_image import GeneratedImage


T = TypeVar("T", bound="ImageGeneration")


@_attrs_define
class ImageGeneration:
    """
    Attributes:
        id (str): This response's own id, for logs. Nothing is stored under it. Example: img_1b9d6bcd-
            bbfd-4b2d-9b5d-ab8dfbbd4bed.
        status (ImageGenerationStatus): Whether the pictures were drawn. A failed generation says why in error_code and
            error.
        images (list[GeneratedImage]): The pictures, as many as were asked for. Empty when the generation failed.
        cost_micros (int): Millionths of a dollar, priced per picture or per megapixel from what came back. Zero when it
            failed.
        provider (str | Unset): Who drew it, or who refused to. Absent when nothing got as far as a provider. Example:
            fal.
        model (str | Unset):  Example: alibaba/qwen-image-3/text-to-image.
        error_code (ImageErrorCode | Unset): Why a generation drew nothing, absent when it completed. content_filtered
            is a safety filter refusing the prompt or the picture; unsupported_option a size, shape or setting no candidate
            could honour; provider_failed anything else a provider did wrong; timeout the 240 seconds running out; cancelled
            the caller hanging up.
        error (str | Unset): What went wrong, in words. Absent when the generation completed.
    """

    id: str
    status: ImageGenerationStatus
    images: list[GeneratedImage]
    cost_micros: int
    provider: str | Unset = UNSET
    model: str | Unset = UNSET
    error_code: ImageErrorCode | Unset = UNSET
    error: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        status = self.status.value

        images = []
        for images_item_data in self.images:
            images_item = images_item_data.to_dict()
            images.append(images_item)

        cost_micros = self.cost_micros

        provider = self.provider

        model = self.model

        error_code: str | Unset = UNSET
        if not isinstance(self.error_code, Unset):
            error_code = self.error_code.value

        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "status": status,
                "images": images,
                "cost_micros": cost_micros,
            }
        )
        if provider is not UNSET:
            field_dict["provider"] = provider
        if model is not UNSET:
            field_dict["model"] = model
        if error_code is not UNSET:
            field_dict["error_code"] = error_code
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.generated_image import GeneratedImage

        d = dict(src_dict)
        id = d.pop("id")

        status = ImageGenerationStatus(d.pop("status"))

        images = []
        _images = d.pop("images")
        for images_item_data in _images:
            images_item = GeneratedImage.from_dict(images_item_data)

            images.append(images_item)

        cost_micros = d.pop("cost_micros")

        provider = d.pop("provider", UNSET)

        model = d.pop("model", UNSET)

        _error_code = d.pop("error_code", UNSET)
        error_code: ImageErrorCode | Unset
        if isinstance(_error_code, Unset):
            error_code = UNSET
        else:
            error_code = ImageErrorCode(_error_code)

        error = d.pop("error", UNSET)

        image_generation = cls(
            id=id,
            status=status,
            images=images,
            cost_micros=cost_micros,
            provider=provider,
            model=model,
            error_code=error_code,
            error=error,
        )

        image_generation.additional_properties = d
        return image_generation

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
