from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.sandbox_limits_network import SandboxLimitsNetwork
from ..types import UNSET, Unset

T = TypeVar("T", bound="SandboxLimits")


@_attrs_define
class SandboxLimits:
    """
    Attributes:
        max_concurrent (int | Unset): How many sandboxes at once.
        max_cpu (int | Unset): The most vCPUs of one sandbox.
        max_disk_gb (int | Unset): The most disk of one sandbox, in GiB.
        max_memory_gb (int | Unset): The most memory of one sandbox, in GiB.
        network (SandboxLimitsNetwork | Unset): What a sandbox may reach: nothing, an allow list, or anything.
    """

    max_concurrent: int | Unset = UNSET
    max_cpu: int | Unset = UNSET
    max_disk_gb: int | Unset = UNSET
    max_memory_gb: int | Unset = UNSET
    network: SandboxLimitsNetwork | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        max_concurrent = self.max_concurrent

        max_cpu = self.max_cpu

        max_disk_gb = self.max_disk_gb

        max_memory_gb = self.max_memory_gb

        network: str | Unset = UNSET
        if not isinstance(self.network, Unset):
            network = self.network.value

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if max_concurrent is not UNSET:
            field_dict["max_concurrent"] = max_concurrent
        if max_cpu is not UNSET:
            field_dict["max_cpu"] = max_cpu
        if max_disk_gb is not UNSET:
            field_dict["max_disk_gb"] = max_disk_gb
        if max_memory_gb is not UNSET:
            field_dict["max_memory_gb"] = max_memory_gb
        if network is not UNSET:
            field_dict["network"] = network

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        max_concurrent = d.pop("max_concurrent", UNSET)

        max_cpu = d.pop("max_cpu", UNSET)

        max_disk_gb = d.pop("max_disk_gb", UNSET)

        max_memory_gb = d.pop("max_memory_gb", UNSET)

        _network = d.pop("network", UNSET)
        network: SandboxLimitsNetwork | Unset
        if isinstance(_network, Unset):
            network = UNSET
        else:
            network = SandboxLimitsNetwork(_network)

        sandbox_limits = cls(
            max_concurrent=max_concurrent,
            max_cpu=max_cpu,
            max_disk_gb=max_disk_gb,
            max_memory_gb=max_memory_gb,
            network=network,
        )

        return sandbox_limits
