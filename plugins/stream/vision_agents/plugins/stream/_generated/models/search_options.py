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
        category (str | Unset): The kind of source to prefer - news, papers, company, github - for the providers that
            classify their index.
        contents (list[SearchOptionsContentsItem] | Unset): What to return alongside each hit.
        depth (SearchDepth | Unset): How much work a search is worth. instant answers from the index in a few hundred
            milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. Providers offer different
            ladders, so each one maps these four onto its own.
        exclude_domains (list[str] | Unset):
        include_domains (list[str] | Unset): Only answer from these domains.
        location (str | Unset): Country or region to answer from, for queries whose answer depends on where.
        max_age_hours (int | Unset): How stale a cached page may be. Zero forces a live crawl, which is slower and costs
            more.
        output_schema (SearchOptionsOutputSchema | Unset): A JSON schema the answer must fit, for the providers that can
            be asked to structure what they found.
        providers (list[str] | Unset): A priority list of where to try, in the order given, which wins over target and
            depth when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded
            where it stands. A search that fails is asked of the next entry that will have it.
             Example: ['exa', 'search-fast'].
        results (int | Unset): How many hits to return.
        target (str | Unset): A provider/model or a capability shortcut. Example: search-fast.
    """

    category: str | Unset = UNSET
    contents: list[SearchOptionsContentsItem] | Unset = UNSET
    depth: SearchDepth | Unset = UNSET
    exclude_domains: list[str] | Unset = UNSET
    include_domains: list[str] | Unset = UNSET
    location: str | Unset = UNSET
    max_age_hours: int | Unset = UNSET
    output_schema: SearchOptionsOutputSchema | Unset = UNSET
    providers: list[str] | Unset = UNSET
    results: int | Unset = UNSET
    target: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        category = self.category

        contents: list[str] | Unset = UNSET
        if not isinstance(self.contents, Unset):
            contents = []
            for contents_item_data in self.contents:
                contents_item = contents_item_data.value
                contents.append(contents_item)

        depth: str | Unset = UNSET
        if not isinstance(self.depth, Unset):
            depth = self.depth.value

        exclude_domains: list[str] | Unset = UNSET
        if not isinstance(self.exclude_domains, Unset):
            exclude_domains = self.exclude_domains

        include_domains: list[str] | Unset = UNSET
        if not isinstance(self.include_domains, Unset):
            include_domains = self.include_domains

        location = self.location

        max_age_hours = self.max_age_hours

        output_schema: dict[str, Any] | Unset = UNSET
        if not isinstance(self.output_schema, Unset):
            output_schema = self.output_schema.to_dict()

        providers: list[str] | Unset = UNSET
        if not isinstance(self.providers, Unset):
            providers = self.providers

        results = self.results

        target = self.target

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if category is not UNSET:
            field_dict["category"] = category
        if contents is not UNSET:
            field_dict["contents"] = contents
        if depth is not UNSET:
            field_dict["depth"] = depth
        if exclude_domains is not UNSET:
            field_dict["exclude_domains"] = exclude_domains
        if include_domains is not UNSET:
            field_dict["include_domains"] = include_domains
        if location is not UNSET:
            field_dict["location"] = location
        if max_age_hours is not UNSET:
            field_dict["max_age_hours"] = max_age_hours
        if output_schema is not UNSET:
            field_dict["output_schema"] = output_schema
        if providers is not UNSET:
            field_dict["providers"] = providers
        if results is not UNSET:
            field_dict["results"] = results
        if target is not UNSET:
            field_dict["target"] = target

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.search_options_output_schema import (
            SearchOptionsOutputSchema,
        )

        d = dict(src_dict)
        category = d.pop("category", UNSET)

        _contents = d.pop("contents", UNSET)
        contents: list[SearchOptionsContentsItem] | Unset = UNSET
        if _contents is not UNSET:
            contents = []
            for contents_item_data in _contents:
                contents_item = SearchOptionsContentsItem(contents_item_data)

                contents.append(contents_item)

        _depth = d.pop("depth", UNSET)
        depth: SearchDepth | Unset
        if isinstance(_depth, Unset):
            depth = UNSET
        else:
            depth = SearchDepth(_depth)

        exclude_domains = cast(list[str], d.pop("exclude_domains", UNSET))

        include_domains = cast(list[str], d.pop("include_domains", UNSET))

        location = d.pop("location", UNSET)

        max_age_hours = d.pop("max_age_hours", UNSET)

        _output_schema = d.pop("output_schema", UNSET)
        output_schema: SearchOptionsOutputSchema | Unset
        if isinstance(_output_schema, Unset):
            output_schema = UNSET
        else:
            output_schema = SearchOptionsOutputSchema.from_dict(_output_schema)

        providers = cast(list[str], d.pop("providers", UNSET))

        results = d.pop("results", UNSET)

        target = d.pop("target", UNSET)

        search_options = cls(
            category=category,
            contents=contents,
            depth=depth,
            exclude_domains=exclude_domains,
            include_domains=include_domains,
            location=location,
            max_age_hours=max_age_hours,
            output_schema=output_schema,
            providers=providers,
            results=results,
            target=target,
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
