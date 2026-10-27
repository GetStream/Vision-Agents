from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from typing_extensions import Self

from ..models.account_balance_state import AccountBalanceState
from ..models.account_balance_warning_source import AccountBalanceWarningSource
from ..models.account_status import AccountStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.limits import Limits
    from ..models.spend_hard_limit_usage import SpendHardLimitUsage


T = TypeVar("T", bound="Account")


@_attrs_define
class Account:
    """The calling organization's account: its balance, its credit and spend, its status and tier, the threshold it is
    warned at, and what Stream allows it. Why an organization is suspended is Stream's, and is not here.

        Attributes:
            balance_micros (int): What the organization has to spend now, in millionths of a dollar: its credit less what it
                spent and what work in flight holds, the figure every answer's X-Stream-Balance-Micros says.
            balance_state (AccountBalanceState): Where balance_micros stands: exhausted at or below 0, low at or below the
                threshold.
            balance_warning_micros (int): The balance at or below which the organization is warned: X-Stream-Balance-Warning
                says low, and its stream event destinations get account.balance_low. 0 when nothing sets one.
            balance_warning_source (AccountBalanceWarningSource): Whose threshold it is: the tier's, the one Stream set for
                this organization, or the organization policy's balance_warning_micros, which wins over both.
            granted_micros (int): Every credit entry summed: top-ups, grants, refunds and adjustments.
            limits (Limits):
            organization_id (str):
            spent_micros (int): Every billable unit of work recorded, summed.
            status (AccountStatus): suspended: Stream refuses every request of the organization, free work included, until
                it restores it.
            tier (str): The tier the organization is billed on.
            own_limits (list[str] | None | Unset): The settings in limits Stream set for this organization rather than
                leaving its tier's, as dotted paths such as channels.voice.concurrent.
            spend_hard_limit (SpendHardLimitUsage | Unset): The most this organization may spend in the UTC day and in the
                calendar month, whatever its credit and its own limits. Set by the operator, never by the customer. A missing
                interval has no limit.
    """

    balance_micros: int
    balance_state: AccountBalanceState
    balance_warning_micros: int
    balance_warning_source: AccountBalanceWarningSource
    granted_micros: int
    limits: Limits
    organization_id: str
    spent_micros: int
    status: AccountStatus
    tier: str
    own_limits: list[str] | None | Unset = UNSET
    spend_hard_limit: SpendHardLimitUsage | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        balance_micros = self.balance_micros

        balance_state = self.balance_state.value

        balance_warning_micros = self.balance_warning_micros

        balance_warning_source = self.balance_warning_source.value

        granted_micros = self.granted_micros

        limits = self.limits.to_dict()

        organization_id = self.organization_id

        spent_micros = self.spent_micros

        status = self.status.value

        tier = self.tier

        own_limits: list[str] | None | Unset
        if isinstance(self.own_limits, Unset):
            own_limits = UNSET
        elif isinstance(self.own_limits, list):
            own_limits = self.own_limits

        else:
            own_limits = self.own_limits

        spend_hard_limit: dict[str, Any] | Unset = UNSET
        if not isinstance(self.spend_hard_limit, Unset):
            spend_hard_limit = self.spend_hard_limit.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "balance_micros": balance_micros,
                "balance_state": balance_state,
                "balance_warning_micros": balance_warning_micros,
                "balance_warning_source": balance_warning_source,
                "granted_micros": granted_micros,
                "limits": limits,
                "organization_id": organization_id,
                "spent_micros": spent_micros,
                "status": status,
                "tier": tier,
            }
        )
        if own_limits is not UNSET:
            field_dict["own_limits"] = own_limits
        if spend_hard_limit is not UNSET:
            field_dict["spend_hard_limit"] = spend_hard_limit

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        from ..models.limits import Limits
        from ..models.spend_hard_limit_usage import SpendHardLimitUsage

        d = dict(src_dict)
        balance_micros = d.pop("balance_micros")

        balance_state = AccountBalanceState(d.pop("balance_state"))

        balance_warning_micros = d.pop("balance_warning_micros")

        balance_warning_source = AccountBalanceWarningSource(
            d.pop("balance_warning_source")
        )

        granted_micros = d.pop("granted_micros")

        limits = Limits.from_dict(d.pop("limits"))

        organization_id = d.pop("organization_id")

        spent_micros = d.pop("spent_micros")

        status = AccountStatus(d.pop("status"))

        tier = d.pop("tier")

        def _parse_own_limits(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                own_limits_type_0 = cast(list[str], data)

                return own_limits_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        own_limits = _parse_own_limits(d.pop("own_limits", UNSET))

        _spend_hard_limit = d.pop("spend_hard_limit", UNSET)
        spend_hard_limit: SpendHardLimitUsage | Unset
        if isinstance(_spend_hard_limit, Unset):
            spend_hard_limit = UNSET
        else:
            spend_hard_limit = SpendHardLimitUsage.from_dict(_spend_hard_limit)

        account = cls(
            balance_micros=balance_micros,
            balance_state=balance_state,
            balance_warning_micros=balance_warning_micros,
            balance_warning_source=balance_warning_source,
            granted_micros=granted_micros,
            limits=limits,
            organization_id=organization_id,
            spent_micros=spent_micros,
            status=status,
            tier=tier,
            own_limits=own_limits,
            spend_hard_limit=spend_hard_limit,
        )

        account.additional_properties = d
        return account

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
