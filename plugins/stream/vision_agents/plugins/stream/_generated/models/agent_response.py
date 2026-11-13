from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_response_status import AgentResponseStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.response_branch import ResponseBranch
    from ..models.rewound_to import RewoundTo


T = TypeVar("T", bound="AgentResponse")


@_attrs_define
class AgentResponse:
    """
    Attributes:
        created_at (datetime.datetime):
        id (str):
        session_id (str):
        status (AgentResponseStatus): cancelled is a turn the caller interrupted, which is a different thing from one
            that failed: nothing went wrong, and what had already been said still counts. refused is a turn the prompt
            injection screen refused: a rewind or a fork carries what it asked forward as withheld.
        error (str | Unset):
        finished_at (datetime.datetime | Unset):
        said (str | Unset): What the person asked, which is the first item of every response.
        branch (ResponseBranch | Unset):
        parent_response_id (str | Unset): The response this one was answered after, absent for a session's first and for
            one answered before responses were kept as a tree.
        rewound_to (RewoundTo | Unset):
    """

    created_at: datetime.datetime
    id: str
    session_id: str
    status: AgentResponseStatus
    error: str | Unset = UNSET
    finished_at: datetime.datetime | Unset = UNSET
    said: str | Unset = UNSET
    branch: ResponseBranch | Unset = UNSET
    parent_response_id: str | Unset = UNSET
    rewound_to: RewoundTo | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at.isoformat()

        id = self.id

        session_id = self.session_id

        status = self.status.value

        error = self.error

        finished_at: str | Unset = UNSET
        if not isinstance(self.finished_at, Unset):
            finished_at = self.finished_at.isoformat()

        said = self.said

        branch: dict[str, Any] | Unset = UNSET
        if not isinstance(self.branch, Unset):
            branch = self.branch.to_dict()

        parent_response_id = self.parent_response_id

        rewound_to: dict[str, Any] | Unset = UNSET
        if not isinstance(self.rewound_to, Unset):
            rewound_to = self.rewound_to.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "created_at": created_at,
                "id": id,
                "session_id": session_id,
                "status": status,
            }
        )
        if error is not UNSET:
            field_dict["error"] = error
        if finished_at is not UNSET:
            field_dict["finished_at"] = finished_at
        if said is not UNSET:
            field_dict["said"] = said
        if branch is not UNSET:
            field_dict["branch"] = branch
        if parent_response_id is not UNSET:
            field_dict["parent_response_id"] = parent_response_id
        if rewound_to is not UNSET:
            field_dict["rewound_to"] = rewound_to

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.response_branch import ResponseBranch
        from ..models.rewound_to import RewoundTo

        d = dict(src_dict)
        created_at = datetime.datetime.fromisoformat(d.pop("created_at"))

        id = d.pop("id")

        session_id = d.pop("session_id")

        status = AgentResponseStatus._decode(d.pop("status"))

        error = d.pop("error", UNSET)

        _finished_at = d.pop("finished_at", UNSET)
        finished_at: datetime.datetime | Unset
        if isinstance(_finished_at, Unset):
            finished_at = UNSET
        else:
            finished_at = datetime.datetime.fromisoformat(_finished_at)

        said = d.pop("said", UNSET)

        _branch = d.pop("branch", UNSET)
        branch: ResponseBranch | Unset
        if isinstance(_branch, Unset):
            branch = UNSET
        else:
            branch = ResponseBranch.from_dict(_branch)

        parent_response_id = d.pop("parent_response_id", UNSET)

        _rewound_to = d.pop("rewound_to", UNSET)
        rewound_to: RewoundTo | Unset
        if isinstance(_rewound_to, Unset):
            rewound_to = UNSET
        else:
            rewound_to = RewoundTo.from_dict(_rewound_to)

        agent_response = cls(
            created_at=created_at,
            id=id,
            session_id=session_id,
            status=status,
            error=error,
            finished_at=finished_at,
            said=said,
            branch=branch,
            parent_response_id=parent_response_id,
            rewound_to=rewound_to,
        )

        agent_response.additional_properties = d
        return agent_response

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
