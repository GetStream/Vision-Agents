from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.budget import Budget
    from ..models.data_policy import DataPolicy


T = TypeVar("T", bound="Policy")


@_attrs_define
class Policy:
    """What an organization or an app decided about spend, data handling and prompt injection. Every field is optional, and
    a field left out is no opinion rather than off.

        Attributes:
            budget (Budget | Unset): A cap on spend across every modality, reset on a UTC boundary each interval. Once it is
                spent every new session and every LLM response is refused until the next interval. Checks are cached for a few
                seconds, so a busy app can overshoot by what it spends in that time.
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            prompt_injection (bool | Unset): Screen what every LLM response is asked for prompt injection. The newest input
                - the user's turn and any tool results - goes to the classifier (lcm) beside the model call, so it adds nothing
                to time to first token. The end of the response is held until the verdict, and a response whose input reads as
                an injection fails with prompt_injection before its tool calls can be acted on.
    """

    budget: Budget | Unset = UNSET
    data_policy: DataPolicy | Unset = UNSET
    prompt_injection: bool | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        budget: dict[str, Any] | Unset = UNSET
        if not isinstance(self.budget, Unset):
            budget = self.budget.to_dict()

        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        prompt_injection = self.prompt_injection

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if budget is not UNSET:
            field_dict["budget"] = budget
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if prompt_injection is not UNSET:
            field_dict["prompt_injection"] = prompt_injection

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.budget import Budget
        from ..models.data_policy import DataPolicy

        d = dict(src_dict)
        _budget = d.pop("budget", UNSET)
        budget: Budget | Unset
        if isinstance(_budget, Unset):
            budget = UNSET
        else:
            budget = Budget.from_dict(_budget)

        _data_policy = d.pop("data_policy", UNSET)
        data_policy: DataPolicy | Unset
        if isinstance(_data_policy, Unset):
            data_policy = UNSET
        else:
            data_policy = DataPolicy.from_dict(_data_policy)

        prompt_injection = d.pop("prompt_injection", UNSET)

        policy = cls(
            budget=budget,
            data_policy=data_policy,
            prompt_injection=prompt_injection,
        )

        policy.additional_properties = d
        return policy

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
