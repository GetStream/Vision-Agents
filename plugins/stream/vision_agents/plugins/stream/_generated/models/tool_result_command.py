from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.tool_result_command_type import ToolResultCommandType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_content_part import ImageContentPart
    from ..models.text_content_part import TextContentPart


T = TypeVar("T", bound="ToolResultCommand")


@_attrs_define
class ToolResultCommand:
    """
    Attributes:
        type_ (ToolResultCommandType):
        tool_call_id (str):
        output (list[ImageContentPart | TextContentPart] | str | Unset):
        error (str | Unset):
    """

    type_: ToolResultCommandType
    tool_call_id: str
    output: list[ImageContentPart | TextContentPart] | str | Unset = UNSET
    error: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.text_content_part import TextContentPart

        type_ = self.type_.value

        tool_call_id = self.tool_call_id

        output: list[dict[str, Any]] | str | Unset
        if isinstance(self.output, Unset):
            output = UNSET
        elif isinstance(self.output, list):
            output = []
            for componentsschemas_message_content_type_1_item_data in self.output:
                componentsschemas_message_content_type_1_item: dict[str, Any]
                if isinstance(
                    componentsschemas_message_content_type_1_item_data, TextContentPart
                ):
                    componentsschemas_message_content_type_1_item = (
                        componentsschemas_message_content_type_1_item_data.to_dict()
                    )
                else:
                    componentsschemas_message_content_type_1_item = (
                        componentsschemas_message_content_type_1_item_data.to_dict()
                    )

                output.append(componentsschemas_message_content_type_1_item)

        else:
            output = self.output

        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "type": type_,
                "tool_call_id": tool_call_id,
            }
        )
        if output is not UNSET:
            field_dict["output"] = output
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.image_content_part import ImageContentPart
        from ..models.text_content_part import TextContentPart

        d = dict(src_dict)
        type_ = ToolResultCommandType(d.pop("type"))

        tool_call_id = d.pop("tool_call_id")

        def _parse_output(
            data: object,
        ) -> list[ImageContentPart | TextContentPart] | str | Unset:
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                componentsschemas_message_content_type_1 = []
                _componentsschemas_message_content_type_1 = data
                for (
                    componentsschemas_message_content_type_1_item_data
                ) in _componentsschemas_message_content_type_1:

                    def _parse_componentsschemas_message_content_type_1_item(
                        data: object,
                    ) -> ImageContentPart | TextContentPart:
                        try:
                            if not isinstance(data, dict):
                                raise TypeError()
                            componentsschemas_content_part_type_0 = (
                                TextContentPart.from_dict(data)
                            )

                            return componentsschemas_content_part_type_0
                        except (TypeError, ValueError, AttributeError, KeyError):
                            pass
                        if not isinstance(data, dict):
                            raise TypeError()
                        componentsschemas_content_part_type_1 = (
                            ImageContentPart.from_dict(data)
                        )

                        return componentsschemas_content_part_type_1

                    componentsschemas_message_content_type_1_item = (
                        _parse_componentsschemas_message_content_type_1_item(
                            componentsschemas_message_content_type_1_item_data
                        )
                    )

                    componentsschemas_message_content_type_1.append(
                        componentsschemas_message_content_type_1_item
                    )

                return componentsschemas_message_content_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[ImageContentPart | TextContentPart] | str | Unset, data)

        output = _parse_output(d.pop("output", UNSET))

        error = d.pop("error", UNSET)

        tool_result_command = cls(
            type_=type_,
            tool_call_id=tool_call_id,
            output=output,
            error=error,
        )

        tool_result_command.additional_properties = d
        return tool_result_command

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
