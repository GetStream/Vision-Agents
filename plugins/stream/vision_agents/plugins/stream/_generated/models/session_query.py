from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.session_filter import SessionFilter
    from ..models.session_sort import SessionSort


T = TypeVar("T", bound="SessionQuery")


@_attrs_define
class SessionQuery:
    """
    Attributes:
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filter and sort. Omitted is the
            first page.
        filter_ (SessionFilter | Unset): Which sessions to list. A field not listed here is refused rather than ignored.
        limit (int | Unset): Up to 200. Omitted is 25.
        sort (list[SessionSort] | None | Unset): Omitted is updated_at, or relevance for a text search.
    """

    cursor: str | Unset = UNSET
    filter_: SessionFilter | Unset = UNSET
    limit: int | Unset = UNSET
    sort: list[SessionSort] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        cursor = self.cursor

        filter_: dict[str, Any] | Unset = UNSET
        if not isinstance(self.filter_, Unset):
            filter_ = self.filter_.to_dict()

        limit = self.limit

        sort: list[dict[str, Any]] | None | Unset
        if isinstance(self.sort, Unset):
            sort = UNSET
        elif isinstance(self.sort, list):
            sort = []
            for sort_type_0_item_data in self.sort:
                sort_type_0_item = sort_type_0_item_data.to_dict()
                sort.append(sort_type_0_item)

        else:
            sort = self.sort

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if cursor is not UNSET:
            field_dict["cursor"] = cursor
        if filter_ is not UNSET:
            field_dict["filter"] = filter_
        if limit is not UNSET:
            field_dict["limit"] = limit
        if sort is not UNSET:
            field_dict["sort"] = sort

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.session_filter import SessionFilter
        from ..models.session_sort import SessionSort

        d = dict(src_dict)
        cursor = d.pop("cursor", UNSET)

        _filter_ = d.pop("filter", UNSET)
        filter_: SessionFilter | Unset
        if isinstance(_filter_, Unset):
            filter_ = UNSET
        else:
            filter_ = SessionFilter.from_dict(_filter_)

        limit = d.pop("limit", UNSET)

        def _parse_sort(data: object) -> list[SessionSort] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                sort_type_0 = []
                _sort_type_0 = data
                for sort_type_0_item_data in _sort_type_0:
                    sort_type_0_item = SessionSort.from_dict(sort_type_0_item_data)

                    sort_type_0.append(sort_type_0_item)

                return sort_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[SessionSort] | None | Unset, data)

        sort = _parse_sort(d.pop("sort", UNSET))

        session_query = cls(
            cursor=cursor,
            filter_=filter_,
            limit=limit,
            sort=sort,
        )

        return session_query
