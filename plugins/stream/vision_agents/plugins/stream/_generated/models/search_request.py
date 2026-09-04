from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.search_options import SearchOptions
    from ..models.search_request_tags import SearchRequestTags


T = TypeVar("T", bound="SearchRequest")


@_attrs_define
class SearchRequest:
    """
    Attributes:
        query (str): The question, in the caller's own words. Example: perioperative antibiotic guidance.
        config_id (str | Unset): A stored router config to take the options from. Anything named here as well overrides
            that one field of it.
        options (SearchOptions | Unset): How this config finds out today's answers.
        tags (SearchRequestTags | Unset):
    """

    query: str
    config_id: str | Unset = UNSET
    options: SearchOptions | Unset = UNSET
    tags: SearchRequestTags | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        query = self.query

        config_id = self.config_id

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
                "query": query,
            }
        )
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if options is not UNSET:
            field_dict["options"] = options
        if tags is not UNSET:
            field_dict["tags"] = tags

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.search_options import SearchOptions
        from ..models.search_request_tags import SearchRequestTags

        d = dict(src_dict)
        query = d.pop("query")

        config_id = d.pop("config_id", UNSET)

        _options = d.pop("options", UNSET)
        options: SearchOptions | Unset
        if isinstance(_options, Unset):
            options = UNSET
        else:
            options = SearchOptions.from_dict(_options)

        _tags = d.pop("tags", UNSET)
        tags: SearchRequestTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = SearchRequestTags.from_dict(_tags)

        search_request = cls(
            query=query,
            config_id=config_id,
            options=options,
            tags=tags,
        )

        search_request.additional_properties = d
        return search_request

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
