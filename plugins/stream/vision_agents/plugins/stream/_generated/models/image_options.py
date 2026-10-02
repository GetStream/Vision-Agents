from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.image_options_output_format import ImageOptionsOutputFormat
from ..types import UNSET, Unset

T = TypeVar("T", bound="ImageOptions")


@_attrs_define
class ImageOptions:
    """Where to draw and what the picture should be. A size, shape, seed, negative prompt or format narrows the candidates
    to the models that declared it, so it is either honoured or the request is refused.

        Attributes:
            aspect_ratio (str | Unset): The shape, for the models that are asked for one rather than a size. Example: 1:1.
            n (int | Unset): How many pictures to draw. Default: 1.
            negative_prompt (str | Unset): What to keep out of the picture.
            output_format (ImageOptionsOutputFormat | Unset): The encoding, on a model that can be asked for one.
            providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target when
                it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it
                stands.
                 Example: ['fal', 'image-quality'].
            seed (int | Unset): Draws the same picture again from the same prompt, on a model that reads one.
            size (str | Unset): Width by height in pixels, for the models that take a size. Example: 1024x1024.
            target (str | Unset): A provider/model or a capability shortcut. Defaults to image-fast. Example: image-fast.
    """

    aspect_ratio: str | Unset = UNSET
    n: int | Unset = 1
    negative_prompt: str | Unset = UNSET
    output_format: ImageOptionsOutputFormat | Unset = UNSET
    providers: list[str] | Unset = UNSET
    seed: int | Unset = UNSET
    size: str | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        aspect_ratio = self.aspect_ratio

        n = self.n

        negative_prompt = self.negative_prompt

        output_format: str | Unset = UNSET
        if not isinstance(self.output_format, Unset):
            output_format = self.output_format.value

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        seed = self.seed

        size = self.size

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if aspect_ratio is not UNSET:
            field_dict["aspect_ratio"] = aspect_ratio
        if n is not UNSET:
            field_dict["n"] = n
        if negative_prompt is not UNSET:
            field_dict["negative_prompt"] = negative_prompt
        if output_format is not UNSET:
            field_dict["output_format"] = output_format
        if providers is not UNSET:
            field_dict["providers"] = providers
        if seed is not UNSET:
            field_dict["seed"] = seed
        if size is not UNSET:
            field_dict["size"] = size
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        aspect_ratio = d.pop("aspect_ratio", UNSET)

        n = d.pop("n", UNSET)

        negative_prompt = d.pop("negative_prompt", UNSET)

        _output_format = d.pop("output_format", UNSET)
        output_format: ImageOptionsOutputFormat | Unset
        if isinstance(_output_format, Unset):
            output_format = UNSET
        else:
            output_format = ImageOptionsOutputFormat(_output_format)

        providers = cast(list[str], d.pop("providers", UNSET))

        seed = d.pop("seed", UNSET)

        size = d.pop("size", UNSET)

        target = d.pop("target", UNSET)

        image_options = cls(
            aspect_ratio=aspect_ratio,
            n=n,
            negative_prompt=negative_prompt,
            output_format=output_format,
            providers=providers,
            seed=seed,
            size=size,
            target=target,
        )

        image_options.additional_properties = d
        return image_options

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
