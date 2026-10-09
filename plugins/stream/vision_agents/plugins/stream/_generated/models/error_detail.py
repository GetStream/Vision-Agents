from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.error_type import ErrorType

T = TypeVar("T", bound="ErrorDetail")


@_attrs_define
class ErrorDetail:
    """
    Attributes:
        code (str): What went wrong, for a program to branch on. Every type has a code of its own name (invalid_request,
            unauthenticated, forbidden, not_found, method_not_allowed, not_acceptable, conflict, gone, payload_too_large,
            unsupported_media_type, rate_limited, internal_error, unavailable) that a failure has when nothing names it
            better. The others are validation_failed, missing_customer, missing_organization, server_side_only,
            not_configured (this deployment does not offer the feature), modality_not_routed, unsynced_changes (a sync asked
            to check would write over somebody's edits), name_taken (a 409: another agent config, router config or voice, or
            another skill of the same agent config, already has the name, so another name will do), and <resource>_not_found
            for agent_config, call, campaign, channel_account, command, connection, knowledge_document, knowledge_url,
            plugin, router_config, session, simulation, simulation_run, skill and voice. More may be added, so a client
            should expect one it does not know.
        doc_url (str): Where the code is explained.
        message (str): What went wrong, for a person to read. Its wording may change; branch on code.
        type_ (ErrorType): The kind of failure, which decides the status it is answered with: invalid_request 400,
            authentication 401, permission 403, not_found 404, method_not_allowed 405, not_acceptable 406, conflict 409,
            gone 410, payload_too_large 413, unsupported_media_type 415, rate_limited 429, internal 500, unavailable 503.
    """

    code: str
    doc_url: str
    message: str
    type_: ErrorType
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        code = self.code

        doc_url = self.doc_url

        message = self.message

        type_ = self.type_.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "code": code,
                "doc_url": doc_url,
                "message": message,
                "type": type_,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        code = d.pop("code")

        doc_url = d.pop("doc_url")

        message = d.pop("message")

        type_ = ErrorType(d.pop("type"))

        error_detail = cls(
            code=code,
            doc_url=doc_url,
            message=message,
            type_=type_,
        )

        error_detail.additional_properties = d
        return error_detail

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
