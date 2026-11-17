from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConnectionValidationRequest")


@_attrs_define
class ConnectionValidationRequest:
    """What a validate checks the grant's scopes against. An unknown field is refused rather than ignored.

    Attributes:
        tools (list[str] | None | Unset): The tools to check the granted scopes against, by name: those an agent config
            will grant. Left out, every tool the connection offers.
    """

    tools: list[str] | None | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        tools: list[str] | None | Unset
        if isinstance(self.tools, Unset):
            tools = UNSET
        elif isinstance(self.tools, list):
            tools = self.tools

        else:
            tools = self.tools

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if tools is not UNSET:
            field_dict["tools"] = tools

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)

        def _parse_tools(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                tools_type_0 = cast(list[str], data)

                return tools_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        tools = _parse_tools(d.pop("tools", UNSET))

        connection_validation_request = cls(
            tools=tools,
        )

        return connection_validation_request
