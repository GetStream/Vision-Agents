from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.llm_options import LlmOptions
    from ..models.router_config_tags import RouterConfigTags
    from ..models.search_options import SearchOptions
    from ..models.stt_options import SttOptions
    from ..models.tts_options import TtsOptions


T = TypeVar("T", bound="RouterConfig")


@_attrs_define
class RouterConfig:
    """
    Attributes:
        id (str):
        name (str):
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        tags (RouterConfigTags | Unset):
        stt (SttOptions | Unset): How this config transcribes, live or from a recording. A field that only means
            something on one of the two forms says so: a recording has no endpointing to do, and a socket has no file to
            write subtitles from. A provider that cannot express a term refuses the request rather than dropping it
            silently.
        tts (TtsOptions | Unset): How this config speaks. A provider that cannot express a term refuses the request
            rather than dropping it silently, since a voice asked to sound urgent and speaking flatly is worse than one that
            says it cannot.
        llm (LlmOptions | Unset): How this config answers. The names are the response parameters the router already
            speaks rather than a second vocabulary for the same things.
        search (SearchOptions | Unset): How this config finds out today's answers.
    """

    id: str
    name: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    tags: RouterConfigTags | Unset = UNSET
    stt: SttOptions | Unset = UNSET
    tts: TtsOptions | Unset = UNSET
    llm: LlmOptions | Unset = UNSET
    search: SearchOptions | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        name = self.name

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        stt: dict[str, Any] | Unset = UNSET
        if not isinstance(self.stt, Unset):
            stt = self.stt.to_dict()

        tts: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tts, Unset):
            tts = self.tts.to_dict()

        llm: dict[str, Any] | Unset = UNSET
        if not isinstance(self.llm, Unset):
            llm = self.llm.to_dict()

        search: dict[str, Any] | Unset = UNSET
        if not isinstance(self.search, Unset):
            search = self.search.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "name": name,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if tags is not UNSET:
            field_dict["tags"] = tags
        if stt is not UNSET:
            field_dict["stt"] = stt
        if tts is not UNSET:
            field_dict["tts"] = tts
        if llm is not UNSET:
            field_dict["llm"] = llm
        if search is not UNSET:
            field_dict["search"] = search

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.llm_options import LlmOptions
        from ..models.router_config_tags import RouterConfigTags
        from ..models.search_options import SearchOptions
        from ..models.stt_options import SttOptions
        from ..models.tts_options import TtsOptions

        d = dict(src_dict)
        id = d.pop("id")

        name = d.pop("name")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        _tags = d.pop("tags", UNSET)
        tags: RouterConfigTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = RouterConfigTags.from_dict(_tags)

        _stt = d.pop("stt", UNSET)
        stt: SttOptions | Unset
        if isinstance(_stt, Unset):
            stt = UNSET
        else:
            stt = SttOptions.from_dict(_stt)

        _tts = d.pop("tts", UNSET)
        tts: TtsOptions | Unset
        if isinstance(_tts, Unset):
            tts = UNSET
        else:
            tts = TtsOptions.from_dict(_tts)

        _llm = d.pop("llm", UNSET)
        llm: LlmOptions | Unset
        if isinstance(_llm, Unset):
            llm = UNSET
        else:
            llm = LlmOptions.from_dict(_llm)

        _search = d.pop("search", UNSET)
        search: SearchOptions | Unset
        if isinstance(_search, Unset):
            search = UNSET
        else:
            search = SearchOptions.from_dict(_search)

        router_config = cls(
            id=id,
            name=name,
            created_at=created_at,
            updated_at=updated_at,
            tags=tags,
            stt=stt,
            tts=tts,
            llm=llm,
            search=search,
        )

        router_config.additional_properties = d
        return router_config

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
