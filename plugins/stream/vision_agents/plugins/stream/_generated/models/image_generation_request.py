from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_generation_request_tags import ImageGenerationRequestTags
    from ..models.image_options import ImageOptions


T = TypeVar("T", bound="ImageGenerationRequest")


@_attrs_define
class ImageGenerationRequest:
    """
    Attributes:
        prompt (str): What to draw, in the caller's own words. Example: A yellow watering can beside a seedling, flat
            illustration, no text.
        options (ImageOptions | Unset): Where to draw and what the picture should be. A size, shape, seed, negative
            prompt or format narrows the candidates to the models that declared it, so it is either honoured or the request
            is refused.
        tags (ImageGenerationRequestTags | Unset):
    """

    prompt: str
    options: ImageOptions | Unset = UNSET
    tags: ImageGenerationRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        prompt = self.prompt

        options: dict[str, Any] | Unset = UNSET
        if not isinstance(self.options, Unset):
            options = self.options.to_dict()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "prompt": prompt,
            }
        )
        if options is not UNSET:
            field_dict["options"] = options
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_generation_request_tags import (
            ImageGenerationRequestTags,
        )
        from ..models.image_options import ImageOptions

        d = dict(src_dict)
        prompt = d.pop("prompt")

        _options = d.pop("options", UNSET)
        options: ImageOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = ImageOptions.from_dict(_options)

        _tags = d.pop("tags", UNSET)
        tags: ImageGenerationRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = ImageGenerationRequestTags.from_dict(_tags)

        image_generation_request = cls(
            prompt=prompt,
            options=options,
            tags=tags,
        )

        image_generation_request.additional_properties = d
        return image_generation_request

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
