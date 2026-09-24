from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.fork_session_request_custom import ForkSessionRequestCustom
    from ..models.model_overwrites import ModelOverwrites


T = TypeVar("T", bound="ForkSessionRequest")


@_attrs_define
class ForkSessionRequest:
    """Continue a conversation as a new one. Everything the parent was opened with is inherited; anything named here is
    written over it, which is what makes a fork useful rather than a copy -- the usual reason to fork is to ask the same
    question of a different model.

        Attributes:
            agent (str | Unset):
            config_id (str | Unset):
            title (str | Unset):
            description (str | Unset):
            project (str | Unset):
            custom (ForkSessionRequestCustom | Unset):
            model_overwrites (ModelOverwrites | Unset): What to change about the models for one session, over whatever its
                agent config decided.
                It is one object rather than a dozen fields at the top level because it is one idea: everything here overrides
                the config, and a caller reading a session back wants to see what they changed in one place rather than diffed
                against a config they would have to fetch. Only the safe knobs are here. Instructions and tools are not, because
                a caller able to rewrite those could make a session impersonate a different agent.
            instructions (str | Unset):
            incognito (bool | Unset): Hold the fork off the record. The parent still exists; this conversation onwards is
                simply not kept.
            messages (bool | Unset): Carry the parent's history into the fork, so the new conversation continues from what
                was already said. False starts the same configuration over from nothing, which is what comparing two answers to
                the same opening question wants.
                 Default: True.
            response_id (str | Unset): Carry the parent's history only up to the end of this response, so the fork continues
                from that point rather than from where the parent is now. The history is read from what the parent recorded,
                which also lets a parent that kept no Chat transcript be forked with its history. Cannot be combined with
                messages false.
            call_id (str | Unset): The call the fork joins. A voice session cannot be forked into a text one or the other
                way about, so this is required when the parent held a call and refused when it did not.
    """

    agent: str | Unset = UNSET
    config_id: str | Unset = UNSET
    title: str | Unset = UNSET
    description: str | Unset = UNSET
    project: str | Unset = UNSET
    custom: ForkSessionRequestCustom | Unset = UNSET
    model_overwrites: ModelOverwrites | Unset = UNSET
    instructions: str | Unset = UNSET
    incognito: bool | Unset = UNSET
    messages: bool | Unset = True
    response_id: str | Unset = UNSET
    call_id: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        agent = self.agent

        config_id = self.config_id

        title = self.title

        description = self.description

        project = self.project

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        model_overwrites: dict[str, Any] | Unset = UNSET
        if not isinstance(self.model_overwrites, Unset):
            model_overwrites = self.model_overwrites.to_dict()

        instructions = self.instructions

        incognito = self.incognito

        messages = self.messages

        response_id = self.response_id

        call_id = self.call_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if agent is not UNSET:
            field_dict["agent"] = agent
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if title is not UNSET:
            field_dict["title"] = title
        if description is not UNSET:
            field_dict["description"] = description
        if project is not UNSET:
            field_dict["project"] = project
        if custom is not UNSET:
            field_dict["custom"] = custom
        if model_overwrites is not UNSET:
            field_dict["model_overwrites"] = model_overwrites
        if instructions is not UNSET:
            field_dict["instructions"] = instructions
        if incognito is not UNSET:
            field_dict["incognito"] = incognito
        if messages is not UNSET:
            field_dict["messages"] = messages
        if response_id is not UNSET:
            field_dict["response_id"] = response_id
        if call_id is not UNSET:
            field_dict["call_id"] = call_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.fork_session_request_custom import (
            ForkSessionRequestCustom,
        )
        from ..models.model_overwrites import ModelOverwrites

        d = dict(src_dict)
        agent = d.pop("agent", UNSET)

        config_id = d.pop("config_id", UNSET)

        title = d.pop("title", UNSET)

        description = d.pop("description", UNSET)

        project = d.pop("project", UNSET)

        _custom = d.pop("custom", UNSET)
        custom: ForkSessionRequestCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = ForkSessionRequestCustom.from_dict(_custom)

        _model_overwrites = d.pop("model_overwrites", UNSET)
        model_overwrites: ModelOverwrites | Unset
        if isinstance(_model_overwrites, Unset):
            model_overwrites = UNSET
        else:
            model_overwrites = ModelOverwrites.from_dict(_model_overwrites)

        instructions = d.pop("instructions", UNSET)

        incognito = d.pop("incognito", UNSET)

        messages = d.pop("messages", UNSET)

        response_id = d.pop("response_id", UNSET)

        call_id = d.pop("call_id", UNSET)

        fork_session_request = cls(
            agent=agent,
            config_id=config_id,
            title=title,
            description=description,
            project=project,
            custom=custom,
            model_overwrites=model_overwrites,
            instructions=instructions,
            incognito=incognito,
            messages=messages,
            response_id=response_id,
            call_id=call_id,
        )

        fork_session_request.additional_properties = d
        return fork_session_request

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
