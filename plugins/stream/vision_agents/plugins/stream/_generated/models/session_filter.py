from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.equals_type_1 import EqualsType1
    from ..models.session_filter_custom import SessionFilterCustom
    from ..models.text_match import TextMatch
    from ..models.time_range import TimeRange


T = TypeVar("T", bound="SessionFilter")


@_attrs_define
class SessionFilter:
    """Which sessions to list. A field not listed here is refused rather than ignored.

    Attributes:
        agent (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        agent_id (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        config_id (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        created_at (TimeRange | Unset):
        custom (SessionFilterCustom | Unset): The session's custom object holds every one of these pairs, which is how a
            caller finds again what it labelled.
        modality (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        project_id (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        state (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
        text (TextMatch | Unset):
        user_id (EqualsType1 | str | Unset): Matches one value exactly: "value" is short for {"$eq": "value"}.
    """

    agent: EqualsType1 | str | Unset = UNSET
    agent_id: EqualsType1 | str | Unset = UNSET
    config_id: EqualsType1 | str | Unset = UNSET
    created_at: TimeRange | Unset = UNSET
    custom: SessionFilterCustom | Unset = UNSET
    modality: EqualsType1 | str | Unset = UNSET
    project_id: EqualsType1 | str | Unset = UNSET
    state: EqualsType1 | str | Unset = UNSET
    text: TextMatch | Unset = UNSET
    user_id: EqualsType1 | str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.equals_type_1 import EqualsType1

        agent: dict[str, Any] | str | Unset
        if isinstance(self.agent, Unset):
            agent = UNSET
        elif isinstance(self.agent, EqualsType1):
            agent = self.agent.to_dict()
        else:
            agent = self.agent

        agent_id: dict[str, Any] | str | Unset
        if isinstance(self.agent_id, Unset):
            agent_id = UNSET
        elif isinstance(self.agent_id, EqualsType1):
            agent_id = self.agent_id.to_dict()
        else:
            agent_id = self.agent_id

        config_id: dict[str, Any] | str | Unset
        if isinstance(self.config_id, Unset):
            config_id = UNSET
        elif isinstance(self.config_id, EqualsType1):
            config_id = self.config_id.to_dict()
        else:
            config_id = self.config_id

        created_at: dict[str, Any] | Unset = UNSET
        if not isinstance(self.created_at, Unset):
            created_at = self.created_at.to_dict()

        custom: dict[str, Any] | Unset = UNSET
        if not isinstance(self.custom, Unset):
            custom = self.custom.to_dict()

        modality: dict[str, Any] | str | Unset
        if isinstance(self.modality, Unset):
            modality = UNSET
        elif isinstance(self.modality, EqualsType1):
            modality = self.modality.to_dict()
        else:
            modality = self.modality

        project_id: dict[str, Any] | str | Unset
        if isinstance(self.project_id, Unset):
            project_id = UNSET
        elif isinstance(self.project_id, EqualsType1):
            project_id = self.project_id.to_dict()
        else:
            project_id = self.project_id

        state: dict[str, Any] | str | Unset
        if isinstance(self.state, Unset):
            state = UNSET
        elif isinstance(self.state, EqualsType1):
            state = self.state.to_dict()
        else:
            state = self.state

        text: dict[str, Any] | Unset = UNSET
        if not isinstance(self.text, Unset):
            text = self.text.to_dict()

        user_id: dict[str, Any] | str | Unset
        if isinstance(self.user_id, Unset):
            user_id = UNSET
        elif isinstance(self.user_id, EqualsType1):
            user_id = self.user_id.to_dict()
        else:
            user_id = self.user_id

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if agent is not UNSET:
            field_dict["agent"] = agent
        if agent_id is not UNSET:
            field_dict["agent_id"] = agent_id
        if config_id is not UNSET:
            field_dict["config_id"] = config_id
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if custom is not UNSET:
            field_dict["custom"] = custom
        if modality is not UNSET:
            field_dict["modality"] = modality
        if project_id is not UNSET:
            field_dict["project_id"] = project_id
        if state is not UNSET:
            field_dict["state"] = state
        if text is not UNSET:
            field_dict["text"] = text
        if user_id is not UNSET:
            field_dict["user_id"] = user_id

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.equals_type_1 import EqualsType1
        from ..models.session_filter_custom import SessionFilterCustom
        from ..models.text_match import TextMatch
        from ..models.time_range import TimeRange

        d = dict(src_dict)

        def _parse_agent(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        agent = _parse_agent(d.pop("agent", UNSET))

        def _parse_agent_id(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        agent_id = _parse_agent_id(d.pop("agent_id", UNSET))

        def _parse_config_id(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        config_id = _parse_config_id(d.pop("config_id", UNSET))

        _created_at = d.pop("created_at", UNSET)
        created_at: TimeRange | Unset
        if isinstance(_created_at, Unset):
            created_at = UNSET
        else:
            created_at = TimeRange.from_dict(_created_at)

        _custom = d.pop("custom", UNSET)
        custom: SessionFilterCustom | Unset
        if isinstance(_custom, Unset):
            custom = UNSET
        else:
            custom = SessionFilterCustom.from_dict(_custom)

        def _parse_modality(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        modality = _parse_modality(d.pop("modality", UNSET))

        def _parse_project_id(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        project_id = _parse_project_id(d.pop("project_id", UNSET))

        def _parse_state(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        state = _parse_state(d.pop("state", UNSET))

        _text = d.pop("text", UNSET)
        text: TextMatch | Unset
        if isinstance(_text, Unset):
            text = UNSET
        else:
            text = TextMatch.from_dict(_text)

        def _parse_user_id(data: object) -> EqualsType1 | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_equals_type_1 = EqualsType1.from_dict(data)

                return componentsschemas_equals_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(EqualsType1 | str | Unset, data)

        user_id = _parse_user_id(d.pop("user_id", UNSET))

        session_filter = cls(
            agent=agent,
            agent_id=agent_id,
            config_id=config_id,
            created_at=created_at,
            custom=custom,
            modality=modality,
            project_id=project_id,
            state=state,
            text=text,
            user_id=user_id,
        )

        return session_filter
