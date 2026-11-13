from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.rewound_to import RewoundTo


T = TypeVar("T", bound="ActivatedBranch")


@_attrs_define
class ActivatedBranch:
    """
    Attributes:
        already_active (bool): The branch was already the conversation, and nothing changed.
        leaf_response_id (str): The newest response of the branch, which the next question is answered after.
        resent (int): How many messages were sent again.
        rewound_to (RewoundTo | Unset):
    """

    already_active: bool
    leaf_response_id: str
    resent: int
    rewound_to: RewoundTo | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        already_active = self.already_active

        leaf_response_id = self.leaf_response_id

        resent = self.resent

        rewound_to: dict[str, Any] | Unset = UNSET
        if not isinstance(self.rewound_to, Unset):
            rewound_to = self.rewound_to.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "already_active": already_active,
                "leaf_response_id": leaf_response_id,
                "resent": resent,
            }
        )
        if rewound_to is not UNSET:
            field_dict["rewound_to"] = rewound_to

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.rewound_to import RewoundTo

        d = dict(src_dict)
        already_active = d.pop("already_active")

        leaf_response_id = d.pop("leaf_response_id")

        resent = d.pop("resent")

        _rewound_to = d.pop("rewound_to", UNSET)
        rewound_to: RewoundTo | Unset
        if isinstance(_rewound_to, Unset):
            rewound_to = UNSET
        else:
            rewound_to = RewoundTo.from_dict(_rewound_to)

        activated_branch = cls(
            already_active=already_active,
            leaf_response_id=leaf_response_id,
            resent=resent,
            rewound_to=rewound_to,
        )

        activated_branch.additional_properties = d
        return activated_branch

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
