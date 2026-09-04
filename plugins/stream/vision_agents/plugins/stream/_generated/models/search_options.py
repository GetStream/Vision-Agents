from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.search_depth import SearchDepth
from ..models.search_options_contents_item import SearchOptionsContentsItem
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.search_options_output_schema import SearchOptionsOutputSchema


T = TypeVar("T", bound="SearchOptions")


@_attrs_define
class SearchOptions:
    """How this config finds out today's answers.

    Attributes:
        target (str | Unset): A provider/model or a capability shortcut. Example: search-fast.
        depth (SearchDepth | Unset): How much work a search is worth. instant answers from the index in a few hundred
            milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different
            ladders, so each one maps these four onto its own.
        results (int | Unset): How many hits to return.
        include_domains (list[str] | Unset): Only answer from these domains.
        exclude_domains (list[str] | Unset):
        category (str | Unset): The kind of source to prefer - news, papers, company, github - for the providers that
            classify their index.
        max_age_hours (int | Unset): How stale a cached page may be. Zero forces a live crawl, which is slower and costs
            more.
        location (str | Unset): Country or region to answer from, for queries whose answer depends on where.
        contents (list[SearchOptionsContentsItem] | Unset): What to return alongside each hit.
        output_schema (SearchOptionsOutputSchema | Unset): A JSON schema the answer must fit, for the providers that can
            be asked to structure what they found.
    """

    target: str | Unset = UNSET
    depth: SearchDepth | Unset = UNSET
    results: int | Unset = UNSET
    include_domains: list[str] | Unset = UNSET
    exclude_domains: list[str] | Unset = UNSET
    category: str | Unset = UNSET
    max_age_hours: int | Unset = UNSET
    location: str | Unset = UNSET
    contents: list[SearchOptionsContentsItem] | Unset = UNSET
    output_schema: SearchOptionsOutputSchema | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        target = self.target

        depth: str | Unset = UNSET
        if not isinstance(self.depth, Unset):
            depth = self.depth.value

        results = self.results

        include_domains: list[str] | Unset = UNSET
        if not isinstance(self.include_domains, Unset):
            include_domains = self.include_domains

        exclude_domains: list[str] | Unset = UNSET
        if not isinstance(self.exclude_domains, Unset):
            exclude_domains = self.exclude_domains

        category = self.category

        max_age_hours = self.max_age_hours

        location = self.location

        contents: list[str] | Unset = UNSET
        if not isinstance(self.contents, Unset):
            contents = []
            for contents_item_data in self.contents:
                contents_item = contents_item_data.value
                contents.append(contents_item)

        output_schema: dict[str, Any] | Unset = UNSET
        if not isinstance(self.output_schema, Unset):
            output_schema = self.output_schema.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if target is not UNSET:
            field_dict["target"] = target
        if depth is not UNSET:
            field_dict["depth"] = depth
        if results is not UNSET:
            field_dict["results"] = results
        if include_domains is not UNSET:
            field_dict["include_domains"] = include_domains
        if exclude_domains is not UNSET:
            field_dict["exclude_domains"] = exclude_domains
        if category is not UNSET:
            field_dict["category"] = category
        if max_age_hours is not UNSET:
            field_dict["max_age_hours"] = max_age_hours
        if location is not UNSET:
            field_dict["location"] = location
        if contents is not UNSET:
            field_dict["contents"] = contents
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.search_options_output_schema import (
            SearchOptionsOutputSchema,
        )

        d = dict(src_dict)
        target = d.pop("target", UNSET)

        _depth = d.pop("depth", UNSET)
        depth: SearchDepth | Unset
        if isinstance(_depth, Unset):
            depth = UNSET
        else:
            depth = SearchDepth(_depth)

        results = d.pop("results", UNSET)

        include_domains = cast(list[str], d.pop("include_domains", UNSET))

        exclude_domains = cast(list[str], d.pop("exclude_domains", UNSET))

        category = d.pop("category", UNSET)

        max_age_hours = d.pop("max_age_hours", UNSET)

        location = d.pop("location", UNSET)

        _contents = d.pop("contents", UNSET)
        contents: list[SearchOptionsContentsItem] | Unset = UNSET
        if _contents is not UNSET:
            contents = []
            for contents_item_data in _contents:
                contents_item = SearchOptionsContentsItem(contents_item_data)

                contents.append(contents_item)

        _output_schema = d.pop("output_schema", UNSET)
        output_schema: SearchOptionsOutputSchema | Unset
        if isinstance(_output_schema, Unset):
            output_schema = UNSET
        else:
            output_schema = SearchOptionsOutputSchema.from_dict(_output_schema)

        search_options = cls(
            target=target,
            depth=depth,
            results=results,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            category=category,
            max_age_hours=max_age_hours,
            location=location,
            contents=contents,
            output_schema=output_schema,
        )

        search_options.additional_properties = d
        return search_options

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
