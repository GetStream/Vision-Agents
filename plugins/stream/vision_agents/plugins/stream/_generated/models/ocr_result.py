from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.ocr_error_code import OcrErrorCode
from ..models.ocr_status import OcrStatus
from ..types import UNSET, Unset

T = TypeVar("T", bound="OcrResult")


@_attrs_define
class OcrResult:
    """
    Attributes:
        cost_micros (int): Millionths of a dollar, priced per page or per token from what the model billed. Zero when it
            failed.
        id (str): This response's own id, for logs. Nothing is stored under it. Example: ocr_1b9d6bcd-
            bbfd-4b2d-9b5d-ab8dfbbd4bed.
        markdown (str): The page's text as Markdown, headings, lists and tables kept. Empty for a page with no text on
            it, or when the reading failed.
        status (OcrStatus): Whether the page was read. A failed reading says why in error_code and error.
        error (str | Unset): What went wrong, in words. Absent when the page was read.
        error_code (OcrErrorCode | Unset): Why a page was not read, absent when it was. provider_failed is every
            candidate failing; timeout the 150 seconds running out; cancelled the caller hanging up.
        model (str | Unset):  Example: mistral-ocr-4-1.
        provider (str | Unset): Who read it, or who failed to. Absent when nothing got as far as a provider. Example:
            mistral.
    """

    cost_micros: int
    id: str
    markdown: str
    status: OcrStatus
    error: str | Unset = UNSET
    error_code: OcrErrorCode | Unset = UNSET
    model: str | Unset = UNSET
    provider: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_micros = self.cost_micros

        id = self.id

        markdown = self.markdown

        status = self.status.value

        error = self.error

        error_code: str | Unset = UNSET
        if not isinstance(self.error_code, Unset):
            error_code = self.error_code.value

        model = self.model

        provider = self.provider

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cost_micros": cost_micros,
                "id": id,
                "markdown": markdown,
                "status": status,
            }
        )
        if error is not UNSET:
            field_dict["error"] = error
        if error_code is not UNSET:
            field_dict["error_code"] = error_code
        if model is not UNSET:
            field_dict["model"] = model
        if provider is not UNSET:
            field_dict["provider"] = provider

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        cost_micros = d.pop("cost_micros")

        id = d.pop("id")

        markdown = d.pop("markdown")

        status = OcrStatus(d.pop("status"))

        error = d.pop("error", UNSET)

        _error_code = d.pop("error_code", UNSET)
        error_code: OcrErrorCode | Unset
        if isinstance(_error_code, Unset):
            error_code = UNSET
        else:
            error_code = OcrErrorCode(_error_code)

        model = d.pop("model", UNSET)

        provider = d.pop("provider", UNSET)

        ocr_result = cls(
            cost_micros=cost_micros,
            id=id,
            markdown=markdown,
            status=status,
            error=error,
            error_code=error_code,
            model=model,
            provider=provider,
        )

        ocr_result.additional_properties = d
        return ocr_result

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
