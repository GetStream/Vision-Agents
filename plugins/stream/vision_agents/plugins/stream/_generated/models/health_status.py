from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.health_status_status import HealthStatusStatus

if TYPE_CHECKING:
    from ..models.health_status_dependencies import HealthStatusDependencies


T = TypeVar("T", bound="HealthStatus")


@_attrs_define
class HealthStatus:
    """
    Attributes:
        status (HealthStatusStatus):
        dependencies (HealthStatusDependencies): Dependency name to "ok" or a failure description. Example: {'postgres':
            'ok', 'redis': 'ok'}.
    """

    status: HealthStatusStatus
    dependencies: HealthStatusDependencies
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status.value

        dependencies = self.dependencies.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "status": status,
                "dependencies": dependencies,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.health_status_dependencies import HealthStatusDependencies

        d = dict(src_dict)
        status = HealthStatusStatus(d.pop("status"))

        dependencies = HealthStatusDependencies.from_dict(d.pop("dependencies"))

        health_status = cls(
            status=status,
            dependencies=dependencies,
        )

        health_status.additional_properties = d
        return health_status

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
