from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.agent_response_item_kind import AgentResponseItemKind
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_response_item_payload import AgentResponseItemPayload


T = TypeVar("T", bound="AgentResponseItem")


@_attrs_define
class AgentResponseItem:
    """
    Attributes:
        at (datetime.datetime):
        kind (AgentResponseItemKind):
        ordinal (int): The position within the response, assigned by the writer rather than by the database, so items
            keep the order they happened in.
        response_id (str):
        payload (AgentResponseItemPayload | Unset): Whatever the kind carries that text cannot: a tool's arguments, a
            guardrail's reason, the id that ties a call to its result.
        session_id (str | Unset):
        text (str | Unset):
        tool_name (str | Unset):
    """

    at: datetime.datetime
    kind: AgentResponseItemKind
    ordinal: int
    response_id: str
    payload: AgentResponseItemPayload | Unset = UNSET
    session_id: str | Unset = UNSET
    text: str | Unset = UNSET
    tool_name: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        at = self.at.isoformat()

        kind = self.kind.value

        ordinal = self.ordinal

        response_id = self.response_id

        payload: dict[str, Any] | Unset = UNSET
        if not isinstance(self.payload, Unset):
            payload = self.payload.to_dict()

        session_id = self.session_id

        text = self.text

        tool_name = self.tool_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "at": at,
                "kind": kind,
                "ordinal": ordinal,
                "response_id": response_id,
            }
        )
        if payload is not UNSET:
            field_dict["payload"] = payload
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if text is not UNSET:
            field_dict["text"] = text
        if tool_name is not UNSET:
            field_dict["tool_name"] = tool_name

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_response_item_payload import AgentResponseItemPayload

        d = dict(src_dict)
        at = datetime.datetime.fromisoformat(d.pop("at"))

        kind = AgentResponseItemKind(d.pop("kind"))

        ordinal = d.pop("ordinal")

        response_id = d.pop("response_id")

        _payload = d.pop("payload", UNSET)
        payload: AgentResponseItemPayload | Unset
        if isinstance(_payload, Unset):
            payload = UNSET
        else:
            payload = AgentResponseItemPayload.from_dict(_payload)

        session_id = d.pop("session_id", UNSET)

        text = d.pop("text", UNSET)

        tool_name = d.pop("tool_name", UNSET)

        agent_response_item = cls(
            at=at,
            kind=kind,
            ordinal=ordinal,
            response_id=response_id,
            payload=payload,
            session_id=session_id,
            text=text,
            tool_name=tool_name,
        )

        agent_response_item.additional_properties = d
        return agent_response_item

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
