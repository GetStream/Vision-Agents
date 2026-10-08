from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.connection_owner_type import ConnectionOwnerType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionOwner")


@_attrs_define
class ConnectionOwner:
    """Whose a connection is: the app's, which any of its agents may be bound to, or one user's.

    Attributes:
        type_ (ConnectionOwnerType): app is the app's own account, user one user's.
        user_id (str | Unset): The user, for a user-owned connection only. It must be the user the backend acts for,
            named by X-Stream-User-Id.
    """

    type_: ConnectionOwnerType
    user_id: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        type_ = self.type_.value

        user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
            }
        )
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        type_ = ConnectionOwnerType(d.pop("type"))

        user_id = d.pop("user_id", UNSET)

        connection_owner = cls(
            type_=type_,
            user_id=user_id,
        )

        return connection_owner
