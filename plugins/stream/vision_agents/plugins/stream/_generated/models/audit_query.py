from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.audit_filter import AuditFilter


T = TypeVar("T", bound="AuditQuery")


@_attrs_define
class AuditQuery:
    """
    Attributes:
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filter. Omitted is the first
            page.
        filter_ (AuditFilter | Unset): Which changes to list. A field not listed here is refused rather than ignored.
        limit (int | Unset): Up to 200. Omitted is 25.
    """

    cursor: str | Unset = UNSET
    filter_: AuditFilter | Unset = UNSET
    limit: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        cursor = self.cursor

        filter_: dict[str, Any] | Unset = UNSET
        if not isinstance(self.filter_, Unset):
            filter_ = self.filter_.to_dict()

        limit = self.limit

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if cursor is not UNSET:
            field_dict["cursor"] = cursor
        if filter_ is not UNSET:
            field_dict["filter"] = filter_
        if limit is not UNSET:
            field_dict["limit"] = limit

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.audit_filter import AuditFilter

        d = dict(src_dict)
        cursor = d.pop("cursor", UNSET)

        _filter_ = d.pop("filter", UNSET)
        filter_: AuditFilter | Unset
        if isinstance(_filter_, Unset):
            filter_ = UNSET
        else:
            filter_ = AuditFilter.from_dict(_filter_)

        limit = d.pop("limit", UNSET)

        audit_query = cls(
            cursor=cursor,
            filter_=filter_,
            limit=limit,
        )

        return audit_query
