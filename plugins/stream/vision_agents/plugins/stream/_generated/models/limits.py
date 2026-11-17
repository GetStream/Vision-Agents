from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.limits_channels import LimitsChannels
    from ..models.sandbox_limits import SandboxLimits
    from ..models.spend_hard_limit import SpendHardLimit


T = TypeVar("T", bound="Limits")


@_attrs_define
class Limits:
    """
    Attributes:
        channels (LimitsChannels | Unset): Limits on each messaging and calling channel, by channel name (sms, whatsapp,
            rcs, imessage, voice, ...).
        low_balance_micros (int | Unset): The balance, in micros, under which the organization is warned.
        sandbox (SandboxLimits | Unset):
        spend_hard_limit (SpendHardLimit | Unset):
    """

    channels: LimitsChannels | Unset = UNSET
    low_balance_micros: int | Unset = UNSET
    sandbox: SandboxLimits | Unset = UNSET
    spend_hard_limit: SpendHardLimit | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        channels: dict[str, Any] | Unset = UNSET
        if not isinstance(self.channels, Unset):
            channels = self.channels.to_dict()

        low_balance_micros = self.low_balance_micros

        sandbox: dict[str, Any] | Unset = UNSET
        if not isinstance(self.sandbox, Unset):
            sandbox = self.sandbox.to_dict()

        spend_hard_limit: dict[str, Any] | Unset = UNSET
        if not isinstance(self.spend_hard_limit, Unset):
            spend_hard_limit = self.spend_hard_limit.to_dict()

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if channels is not UNSET:
            field_dict["channels"] = channels
        if low_balance_micros is not UNSET:
            field_dict["low_balance_micros"] = low_balance_micros
        if sandbox is not UNSET:
            field_dict["sandbox"] = sandbox
        if spend_hard_limit is not UNSET:
            field_dict["spend_hard_limit"] = spend_hard_limit

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.limits_channels import LimitsChannels
        from ..models.sandbox_limits import SandboxLimits
        from ..models.spend_hard_limit import SpendHardLimit

        d = dict(src_dict)
        _channels = d.pop("channels", UNSET)
        channels: LimitsChannels | Unset
        if isinstance(_channels, Unset):
            channels = UNSET
        else:
            channels = LimitsChannels.from_dict(_channels)

        low_balance_micros = d.pop("low_balance_micros", UNSET)

        _sandbox = d.pop("sandbox", UNSET)
        sandbox: SandboxLimits | Unset
        if isinstance(_sandbox, Unset):
            sandbox = UNSET
        else:
            sandbox = SandboxLimits.from_dict(_sandbox)

        _spend_hard_limit = d.pop("spend_hard_limit", UNSET)
        spend_hard_limit: SpendHardLimit | Unset
        if isinstance(_spend_hard_limit, Unset):
            spend_hard_limit = UNSET
        else:
            spend_hard_limit = SpendHardLimit.from_dict(_spend_hard_limit)

        limits = cls(
            channels=channels,
            low_balance_micros=low_balance_micros,
            sandbox=sandbox,
            spend_hard_limit=spend_hard_limit,
        )

        return limits
