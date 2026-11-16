from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="PolicyDataPolicy")


@_attrs_define
class PolicyDataPolicy:
    """What an organization or an app requires of what happens to what its requests send. It is a floor every routed
    request is held to, on top of whatever the request asks: a request is only routed to a model whose declared handling
    meets both, and is refused rather than sent somewhere that does not. It says nothing of where the work is done,
    which is an agent's own region (agent.yaml's region) or what a direct call's own data_policy.region asks, so region
    is refused here, as is any other field it does not have.

        Attributes:
            allow_training (bool | Unset): False requires a provider that has said it does not train on what it is sent.
                Omitting this asks nothing. A provider that has published nothing either way counts as not having said no.
            retention (str | Unset): The longest a provider may keep what it is sent - none, or a duration such as 30d or
                24h. Omitting it asks nothing. Example: none.
    """

    allow_training: bool | Unset = UNSET
    retention: str | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        allow_training = self.allow_training

        retention = self.retention

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if allow_training is not UNSET:
            field_dict["allow_training"] = allow_training
        if retention is not UNSET:
            field_dict["retention"] = retention

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allow_training = d.pop("allow_training", UNSET)

        retention = d.pop("retention", UNSET)

        policy_data_policy = cls(
            allow_training=allow_training,
            retention=retention,
        )

        return policy_data_policy
