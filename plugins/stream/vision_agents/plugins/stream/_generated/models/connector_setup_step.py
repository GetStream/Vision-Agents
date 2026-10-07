from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="ConnectorSetupStep")


@_attrs_define
class ConnectorSetupStep:
    """One step of a provider's setup.

    Attributes:
        description (str):
        title (str):
    """

    description: str
    title: str

    def to_dict(self) -> dict[str, Any]:
        description = self.description

        title = self.title

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "description": description,
                "title": title,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        description = d.pop("description")

        title = d.pop("title")

        connector_setup_step = cls(
            description=description,
            title=title,
        )

        return connector_setup_step
