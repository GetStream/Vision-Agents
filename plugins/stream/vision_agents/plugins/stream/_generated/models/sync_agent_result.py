from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_config import AgentConfig
    from ..models.sync_change import SyncChange


T = TypeVar("T", bound="SyncAgentResult")


@_attrs_define
class SyncAgentResult:
    """
    Attributes:
        config (AgentConfig):
        unchanged (bool): True when the hash matched and nothing was written.
        warnings (list[str] | None | Unset): What was stored but will not work yet, such as a channel line the app has
            not connected.
        changes (list[SyncChange] | Unset): What the sync did, or with dry_run would do, to what the agent holds beside
            its config: each skill, knowledge document, knowledge url and simulation it created, updated or deleted. One it
            leaves as it is is not listed, and nothing is for a sync that found the hash unchanged.
    """

    config: AgentConfig
    unchanged: bool
    warnings: list[str] | None | Unset = UNSET
    changes: list[SyncChange] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        config = self.config.to_dict()

        unchanged = self.unchanged

        warnings: list[str] | None | Unset
        if isinstance(self.warnings, Unset):
            warnings = UNSET
        elif isinstance(self.warnings, list):
            warnings = self.warnings

        else:
            warnings = self.warnings

        changes: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.changes, Unset):
            changes = []
            for changes_item_data in self.changes:
                changes_item = changes_item_data.to_dict()
                changes.append(changes_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "config": config,
                "unchanged": unchanged,
            }
        )
        if warnings is not UNSET:
            field_dict["warnings"] = warnings
        if changes is not UNSET:
            field_dict["changes"] = changes

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.agent_config import AgentConfig
        from ..models.sync_change import SyncChange

        d = dict(src_dict)
        config = AgentConfig.from_dict(d.pop("config"))

        unchanged = d.pop("unchanged")

        def _parse_warnings(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                warnings_type_0 = cast(list[str], data)

                return warnings_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        warnings = _parse_warnings(d.pop("warnings", UNSET))

        _changes = d.pop("changes", UNSET)
        changes: list[SyncChange] | Unset = UNSET
        if _changes is not UNSET:
            changes = []
            for changes_item_data in _changes:
                changes_item = SyncChange.from_dict(changes_item_data)

                changes.append(changes_item)

        sync_agent_result = cls(
            config=config,
            unchanged=unchanged,
            warnings=warnings,
            changes=changes,
        )

        sync_agent_result.additional_properties = d
        return sync_agent_result

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
