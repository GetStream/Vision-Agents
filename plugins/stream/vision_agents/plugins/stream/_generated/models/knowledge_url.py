from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.knowledge_url_state import KnowledgeUrlState
from ..types import UNSET, Unset

T = TypeVar("T", bound="KnowledgeUrl")


@_attrs_define
class KnowledgeUrl:
    """
    Attributes:
        id (str):
        namespace (str):
        url (str):
        state (KnowledgeUrlState): Where the page has got to. Pending means it has been added but not yet read, which is
            also what a read that died halfway through leaves behind.
        passages (int): How many passages the page was last cut into.
        created_at (datetime.datetime):
        updated_at (datetime.datetime):
        title (str | Unset): What the page called itself when it was last read.
        error (str | Unset): Why the last read failed. Empty otherwise.
        last_indexed_at (datetime.datetime | None | Unset): When it was last read successfully. Absent means never,
            which is what separates a page that has never worked from one that worked and has since broken.
    """

    id: str
    namespace: str
    url: str
    state: KnowledgeUrlState
    passages: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    title: str | Unset = UNSET
    error: str | Unset = UNSET
    last_indexed_at: datetime.datetime | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        id = self.id

        namespace = self.namespace

        url = self.url

        state = self.state.value

        passages = self.passages

        created_at = self.created_at.isoformat()

        updated_at = self.updated_at.isoformat()

        title = self.title

        error = self.error

        last_indexed_at: None | str | Unset
        if isinstance(self.last_indexed_at, Unset):
            last_indexed_at = UNSET
        elif isinstance(self.last_indexed_at, datetime.datetime):
            last_indexed_at = self.last_indexed_at.isoformat()
        else:
            last_indexed_at = self.last_indexed_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "id": id,
                "namespace": namespace,
                "url": url,
                "state": state,
                "passages": passages,
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )
        if title is not UNSET:
            field_dict["title"] = title
        if error is not UNSET:
            field_dict["error"] = error
        if last_indexed_at is not UNSET:
            field_dict["last_indexed_at"] = last_indexed_at

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        id = d.pop("id")

        namespace = d.pop("namespace")

        url = d.pop("url")

        state = KnowledgeUrlState(d.pop("state"))

        passages = d.pop("passages")

        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        title = d.pop("title", UNSET)

        error = d.pop("error", UNSET)

        def _parse_last_indexed_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                last_indexed_at_type_0 = datetime.datetime.fromisoformat(data)

                return last_indexed_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        last_indexed_at = _parse_last_indexed_at(d.pop("last_indexed_at", UNSET))

        knowledge_url = cls(
            id=id,
            namespace=namespace,
            url=url,
            state=state,
            passages=passages,
            created_at=created_at,
            updated_at=updated_at,
            title=title,
            error=error,
            last_indexed_at=last_indexed_at,
        )

        knowledge_url.additional_properties = d
        return knowledge_url

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
