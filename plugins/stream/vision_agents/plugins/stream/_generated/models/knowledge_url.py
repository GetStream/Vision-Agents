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
        created_at (datetime.datetime):
        id (str):
        namespace (str):
        passages (int): How many passages the page was last cut into.
        state (KnowledgeUrlState): Where the page has got to. Pending means it has been added and its first read is
            queued or being retried; failed means every attempt failed.
        updated_at (datetime.datetime):
        url (str):
        description (str | Unset): What the page was subscribed as being. Empty unless it was given one.
        error (str | Unset): Why the last read failed. Empty otherwise.
        last_indexed_at (datetime.datetime | None | Unset): When it was last read successfully. Absent means never,
            which is what separates a page that has never worked from one that worked and has since broken.
        refresh_hours (int | Unset): How often the page is read again on its own, in hours. Absent means never.
        title (str | Unset): What the page is called: the title it was subscribed with, or what it called itself when it
            was last read.
    """

    created_at: datetime.datetime
    id: str
    namespace: str
    passages: int
    state: KnowledgeUrlState
    updated_at: datetime.datetime
    url: str
    description: str | Unset = UNSET
    error: str | Unset = UNSET
    last_indexed_at: datetime.datetime | None | Unset = UNSET
    refresh_hours: int | Unset = UNSET
    title: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        namespace = self.namespace

        passages = self.passages

        state = self.state.value

        updated_at = self.updated_at.isoformat()

        url = self.url

        description = self.description

        error = self.error

        last_indexed_at: None | str | Unset
        if isinstance(self.last_indexed_at, Unset):
            last_indexed_at = UNSET
        elif isinstance(self.last_indexed_at, datetime.datetime):
            last_indexed_at = self.last_indexed_at.isoformat()
        else:
            last_indexed_at = self.last_indexed_at

        refresh_hours = self.refresh_hours

        title = self.title

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "namespace": namespace,
                "passages": passages,
                "state": state,
                "updated_at": updated_at,
                "url": url,
            }
        )
        if description is not UNSET:
            field_dict["description"] = description
        if error is not UNSET:
            field_dict["error"] = error
        if last_indexed_at is not UNSET:
            field_dict["last_indexed_at"] = last_indexed_at
        if refresh_hours is not UNSET:
            field_dict["refresh_hours"] = refresh_hours
        if title is not UNSET:
            field_dict["title"] = title

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        namespace = d.pop("namespace")

        passages = d.pop("passages")

        state = KnowledgeUrlState(d.pop("state"))

        updated_at = datetime.datetime.fromisoformat(d.pop("updated_at"))

        url = d.pop("url")

        description = d.pop("description", UNSET)

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

        refresh_hours = d.pop("refresh_hours", UNSET)

        title = d.pop("title", UNSET)

        knowledge_url = cls(
            created_at=created_at,
            id=id,
            namespace=namespace,
            passages=passages,
            state=state,
            updated_at=updated_at,
            url=url,
            description=description,
            error=error,
            last_indexed_at=last_indexed_at,
            refresh_hours=refresh_hours,
            title=title,
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
