from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.connector_setup_step import ConnectorSetupStep


T = TypeVar("T", bound="ConnectorSetup")


@_attrs_define
class ConnectorSetup:
    """What a person does at the provider before the first consent.

    Attributes:
        steps (list[ConnectorSetupStep]): In order.
        url (str | Unset): Where the steps start, a page of the provider's.
    """

    steps: list[ConnectorSetupStep]
    url: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        steps = []
        for steps_item_data in self.steps:
            steps_item = steps_item_data.to_dict()
            steps.append(steps_item)

        url = self.url

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "steps": steps,
            }
        )
        if url is not UNSET:
            field_dict["url"] = url

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.connector_setup_step import ConnectorSetupStep

        d = dict(src_dict)
        steps = []
        _steps = d.pop("steps")
        for steps_item_data in _steps:
            steps_item = ConnectorSetupStep.from_dict(steps_item_data)

            steps.append(steps_item)

        url = d.pop("url", UNSET)

        connector_setup = cls(
            steps=steps,
            url=url,
        )

        return connector_setup
