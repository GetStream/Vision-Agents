from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.budget import Budget
    from ..models.data_policy import DataPolicy
    from ..models.policy_tags import PolicyTags
    from ..models.quota import Quota


T = TypeVar("T", bound="Policy")


@_attrs_define
class Policy:
    """What an organization or an app decided about spend, data handling, prompt injection, which models may be used and
    how usage is labelled. Every field is optional, and a field left out is no opinion rather than off.

        Attributes:
            allowed_models (list[str] | Unset): The only models requests may be routed to, as provider/model names, in every
                modality. Left out allows every model, and an empty list allows none. A request that could only go to models not
                on the list is refused, and a failover never reaches one. Example: ['deepseek/DeepSeek-V4-Flash-0731'].
            budget (Budget | Unset): A cap on spend across every modality, reset on a UTC boundary each interval. Once it is
                spent every new session and every LLM response is refused until the next interval. Checks are cached for a few
                seconds, so a busy app can overshoot by what it spends in that time.
            data_policy (DataPolicy | Unset): What a caller requires of what happens to what they send: the audio they had
                transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a
                description: a request naming one is only routed to a model whose declared handling meets it, and if none does
                the request is refused rather than sent somewhere that does not.
            prompt_injection (bool | Unset): Screen what every LLM response is asked for prompt injection. The newest input
                - the user's turn and any tool results - goes to a decision model beside the model call, so it adds nothing to
                time to first token. The end of the response is held until the verdict, and a response whose input reads as an
                injection fails with prompt_injection before its tool calls can be acted on.
            quota (Quota | Unset): What one end user of the app may spend in a UTC day, counted per user and per address.
                Only an app's policy has one: an organization's that sets it is refused. A turn is counted on the day of the
                user who sent it, and a backend's own are never counted. A request over the limit is a 429 rate_limited with
                Retry-After, and a counted request carries X-Quota-Limit, X-Quota-Remaining and X-Quota-Reset.
            require_own_stream_app (bool | Unset): Keep the app out of the router's own Stream app: in app mode it is never
                written there for want of a registered Stream app of its own, and what it wrote there before can only be read.
                True at either scope requires it, so an app cannot turn its organization's off. An organization's is set by the
                router's operator and read here; sending it back unchanged is fine, and changing it is refused.
            tags (PolicyTags | Unset): Labels recorded on every row of usage, over whatever the request labelled it with, so
                spend is attributed whatever a caller sends. Together with the request's own they must fit in 16 tags. Example:
                {'application': 'support'}.
            tool_output_retention (str | Unset): How long a session keeps what its tools were given and gave back on the
                turns it records: a tool's arguments, its result, a search's query and the wording of a tool's error. none
                records none of it; a duration such as 12h or 30d (at least 1h, at most 3650d) keeps it that long after the call
                and then takes it out, within minutes. The item stays, with its tool name, its time, its call id and whether it
                failed, so a session's items read in the same order and tool usage still counts the call. Left out, or empty, it
                is kept as long as the session. The shortest of the organization's, the app's and the agent config's
                tools.output_retention applies, so an agent can shorten it and never lengthen it. It applies to tool calls
                recorded after it is set. Deleting a session deletes all of it at once. Example: 30d.
    """

    allowed_models: list[str] | Unset = UNSET
    budget: Budget | Unset = UNSET
    data_policy: DataPolicy | Unset = UNSET
    prompt_injection: bool | Unset = UNSET
    quota: Quota | Unset = UNSET
    require_own_stream_app: bool | Unset = UNSET
    tags: PolicyTags | Unset = UNSET
    tool_output_retention: str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        allowed_models: list[str] | Unset = UNSET
        if not isinstance(self.allowed_models, Unset):
            allowed_models = self.allowed_models

        budget: dict[str, Any] | Unset = UNSET
        if not isinstance(self.budget, Unset):
            budget = self.budget.to_dict()

        data_policy: dict[str, Any] | Unset = UNSET
        if not isinstance(self.data_policy, Unset):
            data_policy = self.data_policy.to_dict()

        prompt_injection = self.prompt_injection

        quota: dict[str, Any] | Unset = UNSET
        if not isinstance(self.quota, Unset):
            quota = self.quota.to_dict()

        require_own_stream_app = self.require_own_stream_app

        tags: dict[str, Any] | Unset = UNSET
        if not isinstance(self.tags, Unset):
            tags = self.tags.to_dict()

        tool_output_retention = self.tool_output_retention

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if allowed_models is not UNSET:
            field_dict["allowed_models"] = allowed_models
        if budget is not UNSET:
            field_dict["budget"] = budget
        if data_policy is not UNSET:
            field_dict["data_policy"] = data_policy
        if prompt_injection is not UNSET:
            field_dict["prompt_injection"] = prompt_injection
        if quota is not UNSET:
            field_dict["quota"] = quota
        if require_own_stream_app is not UNSET:
            field_dict["require_own_stream_app"] = require_own_stream_app
        if tags is not UNSET:
            field_dict["tags"] = tags
        if tool_output_retention is not UNSET:
            field_dict["tool_output_retention"] = tool_output_retention

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.budget import Budget
        from ..models.data_policy import DataPolicy
        from ..models.policy_tags import PolicyTags
        from ..models.quota import Quota

        d = dict(src_dict)
        allowed_models = cast(list[str], d.pop("allowed_models", UNSET))

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

        _quota = d.pop("quota", UNSET)
        quota: Quota | Unset
        if isinstance(_quota, Unset):
            quota = UNSET
        else:
            quota = Quota.from_dict(_quota)

        require_own_stream_app = d.pop("require_own_stream_app", UNSET)

        _tags = d.pop("tags", UNSET)
        tags: PolicyTags | Unset
        if isinstance(_tags, Unset):
            tags = UNSET
        else:
            tags = PolicyTags.from_dict(_tags)

        tool_output_retention = d.pop("tool_output_retention", UNSET)

        policy = cls(
            allowed_models=allowed_models,
            budget=budget,
            data_policy=data_policy,
            prompt_injection=prompt_injection,
            quota=quota,
            require_own_stream_app=require_own_stream_app,
            tags=tags,
            tool_output_retention=tool_output_retention,
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
