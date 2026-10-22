from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_run_request_tags import CreateRunRequestTags
    from ..models.image_source import ImageSource


T = TypeVar("T", bound="CreateRunRequest")


@_attrs_define
class CreateRunRequest:
    """
    Attributes:
        config_id (str): The agent config to run.
        input_ (Any): What the agent works on: a string, or any JSON value. It is handed to the agent as data, never as
            instructions.
        images (list[ImageSource] | Unset): Images that go with the input, in order, so the input can refer to them by
            their place in this list. At most 8. An agent with no vision skill and a conversation model that cannot see
            refuses them with a 400 invalid_config.
        instructions (str | Unset): What the agent is told for this run alone, added to its config's instructions.
        tags (CreateRunRequestTags | Unset): Cost labels, carried onto every request the run makes alongside the
            config's own.
        timeout_ms (int | Unset): How long the run may take before it is given up as a 504. Two minutes when left out.
    """

    config_id: str
    input_: Any
    images: list[ImageSource] | Unset = UNSET
    instructions: str | Unset = UNSET
    tags: CreateRunRequestTags | Unset = UNSET
    timeout_ms: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config_id = self.config_id

        input_ = self.input_

        images: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.images, Unset):
            images = []
            for images_item_data in self.images:
                images_item = images_item_data.to_dict()
                images.append(images_item)

        instructions = self.instructions

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        timeout_ms = self.timeout_ms

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config_id": config_id,
                "input": input_,
            }
        )
        if images is not UNSET:
            field_dict["images"] = images
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if tags is not UNSET:
            field_dict["tags"] = tags
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.create_run_request_tags import CreateRunRequestTags
        from ..models.image_source import ImageSource

        d = dict(src_dict)
        config_id = d.pop("config_id")

        input_ = d.pop("input")

        _images = d.pop("images", UNSET)
        images: list[ImageSource] | Unset = UNSET
        if _images is not UNSET:
            images = []
            for images_item_data in _images:
                images_item = ImageSource.from_dict(images_item_data)

                images.append(images_item)

        instructions = d.pop("instructions", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: CreateRunRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = CreateRunRequestTags.from_dict(_tags)

        timeout_ms = d.pop("timeout_ms", UNSET)

        create_run_request = cls(
            config_id=config_id,
            input_=input_,
            images=images,
            instructions=instructions,
            tags=tags,
            timeout_ms=timeout_ms,
        )

        create_run_request.additional_properties = d
        return create_run_request

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
