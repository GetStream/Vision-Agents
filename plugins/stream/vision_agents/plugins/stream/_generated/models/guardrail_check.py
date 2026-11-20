from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.guardrail_check_kind import GuardrailCheckKind
from ..models.guardrail_check_mode import GuardrailCheckMode
from ..types import UNSET, Unset

T = TypeVar("T", bound="GuardrailCheck")


@_attrs_define
class GuardrailCheck:
    """
    Attributes:
        allowed (bool): Whether the turn was answered. True for a check that failed, whose turn is answered anyway.
        kind (GuardrailCheckKind): How the turn was checked: the guardrail.md's type.
        latency_ms (float): How long the check took, from asking to the verdict or the failure.
        mode (GuardrailCheckMode): When the check ran: parallel, beside the model, or blocking, before the model was
            asked.
        failed (bool | Unset): True when the check could not be made (the model or the webhook failed or took too long)
            and the turn was answered anyway. Such a check has no probability. Absent for a check that answered.
        held_ms (float | Unset): How long a reply the model had begun waited on the verdict, which is what a parallel
            check costs the caller. Absent when it did not wait, and in blocking mode, where the model is asked only after
            the verdict.
        model (str | Unset): The provider and model that answered, as provider/model. Absent for a webhook, and for a
            check that failed.
        probability (float | Unset): The probability the model put on the turn going against the policy. Absent for a
            webhook, which answers yes or no, and for a check that failed.
        threshold (float | Unset): The guardrail.md's threshold: a probability at or above it refuses the turn. Absent
            for a webhook.
    """

    allowed: bool
    kind: GuardrailCheckKind
    latency_ms: float
    mode: GuardrailCheckMode
    failed: bool | Unset = UNSET
    held_ms: float | Unset = UNSET
    model: str | Unset = UNSET
    probability: float | Unset = UNSET
    threshold: float | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed = self.allowed

        kind = self.kind.value

        latency_ms = self.latency_ms

        mode = self.mode.value

        failed = self.failed

        held_ms = self.held_ms

        model = self.model

        probability = self.probability

        threshold = self.threshold

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "allowed": allowed,
                "kind": kind,
                "latency_ms": latency_ms,
                "mode": mode,
            }
        )
        if failed is not UNSET:
            field_dict["failed"] = failed
        if held_ms is not UNSET:
            field_dict["held_ms"] = held_ms
        if model is not UNSET:
            field_dict["model"] = model
        if probability is not UNSET:
            field_dict["probability"] = probability
        if threshold is not UNSET:
            field_dict["threshold"] = threshold

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allowed = d.pop("allowed")

        kind = GuardrailCheckKind._decode(d.pop("kind"))

        latency_ms = d.pop("latency_ms")

        mode = GuardrailCheckMode._decode(d.pop("mode"))

        failed = d.pop("failed", UNSET)

        held_ms = d.pop("held_ms", UNSET)

        model = d.pop("model", UNSET)

        probability = d.pop("probability", UNSET)

        threshold = d.pop("threshold", UNSET)

        guardrail_check = cls(
            allowed=allowed,
            kind=kind,
            latency_ms=latency_ms,
            mode=mode,
            failed=failed,
            held_ms=held_ms,
            model=model,
            probability=probability,
            threshold=threshold,
        )

        guardrail_check.additional_properties = d
        return guardrail_check

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
