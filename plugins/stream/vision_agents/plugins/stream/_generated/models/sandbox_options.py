from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..models.sandbox_options_region import SandboxOptionsRegion
from ..models.sandbox_options_snapshot import SandboxOptionsSnapshot
from ..types import UNSET, Unset

T = TypeVar("T", bound="SandboxOptions")


@_attrs_define
class SandboxOptions:
    """How the sandbox is built, where it runs, what it may reach and how long code may run in it. Only meaningful with a
    sandbox. Omit it for the provider's own Python sandbox in its default region, no network and a 30 second run.

        Attributes:
            allowed_domains (list[str] | Unset): Domains code in the sandbox may reach, such as api.example.com. Set,
                nothing else is reachable, even with network on. Not with network_allow_list.
            cpu (int | Unset): CPUs for the sandbox. Zero is the provider's default.
            disk_gb (int | Unset): Disk for the sandbox, in GiB. Zero is the provider's default.
            image (str | Unset): The container image to build on, which must have Python, such as python:3.13-slim-bookworm.
                Empty with anything else set is a slim Python 3.13 image.
            memory_gb (int | Unset): Memory for the sandbox, in GiB. Zero is the provider's default.
            network (bool | Unset): Whether code in the sandbox may reach the internet. Off by default, and then every
                outbound connection is blocked unless allowed_domains or network_allow_list names where it may go.
            network_allow_list (list[str] | Unset): CIDR ranges code in the sandbox may reach, such as 10.0.0.0/8. Set,
                nothing else is reachable, even with network on. Not with allowed_domains.
            region (SandboxOptionsRegion | Unset): Where the sandbox runs: us for the United States, eu for Europe. Empty is
                the provider's default.
            setup (list[str] | Unset): Shell commands run once on top of the image when it is built, such as installing
                packages. The provider keeps the built image, so only the first sandbox from a given setup waits for it.
            snapshot (SandboxOptionsSnapshot | Unset): One of the provider's own Python sandboxes, at its own size: daytona-
                small (1 CPU, 1 GiB memory, 3 GiB disk), daytona-medium (2, 4, 8), daytona-large (4, 8, 10), or a virtual
                machine of the same sizes, daytona-vm-small, daytona-vm-medium or daytona-vm-large. It is built and sized
                already, so not with image, setup, cpu, memory_gb or disk_gb. Empty is the provider's default.
            timeout_ms (int | Unset): How long one run of code may take, at most 30 minutes. Zero is 30 seconds. A run is
                still bounded by the deadline of the skill it was written for.
    """

    allowed_domains: list[str] | Unset = UNSET
    cpu: int | Unset = UNSET
    disk_gb: int | Unset = UNSET
    image: str | Unset = UNSET
    memory_gb: int | Unset = UNSET
    network: bool | Unset = UNSET
    network_allow_list: list[str] | Unset = UNSET
    region: SandboxOptionsRegion | Unset = UNSET
    setup: list[str] | Unset = UNSET
    snapshot: SandboxOptionsSnapshot | Unset = UNSET
    timeout_ms: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        allowed_domains: list[str] | Unset = UNSET
        if not isinstance(self.allowed_domains, Unset):
            allowed_domains = self.allowed_domains

        cpu = self.cpu

        disk_gb = self.disk_gb

        image = self.image

        memory_gb = self.memory_gb

        network = self.network

        network_allow_list: list[str] | Unset = UNSET
        if not isinstance(self.network_allow_list, Unset):
            network_allow_list = self.network_allow_list

        region: str | Unset = UNSET
        if not isinstance(self.region, Unset):
            region = self.region.value

        setup: list[str] | Unset = UNSET
        if not isinstance(self.setup, Unset):
            setup = self.setup

        snapshot: str | Unset = UNSET
        if not isinstance(self.snapshot, Unset):
            snapshot = self.snapshot.value

        timeout_ms = self.timeout_ms

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if allowed_domains is not UNSET:
            field_dict["allowed_domains"] = allowed_domains
        if cpu is not UNSET:
            field_dict["cpu"] = cpu
        if disk_gb is not UNSET:
            field_dict["disk_gb"] = disk_gb
        if image is not UNSET:
            field_dict["image"] = image
        if memory_gb is not UNSET:
            field_dict["memory_gb"] = memory_gb
        if network is not UNSET:
            field_dict["network"] = network
        if network_allow_list is not UNSET:
            field_dict["network_allow_list"] = network_allow_list
        if region is not UNSET:
            field_dict["region"] = region
        if setup is not UNSET:
            field_dict["setup"] = setup
        if snapshot is not UNSET:
            field_dict["snapshot"] = snapshot
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        allowed_domains = cast(list[str], d.pop("allowed_domains", UNSET))

        cpu = d.pop("cpu", UNSET)

        disk_gb = d.pop("disk_gb", UNSET)

        image = d.pop("image", UNSET)

        memory_gb = d.pop("memory_gb", UNSET)

        network = d.pop("network", UNSET)

        network_allow_list = cast(list[str], d.pop("network_allow_list", UNSET))

        _region = d.pop("region", UNSET)
        region: SandboxOptionsRegion | Unset
        if isinstance(_region, Unset):
            region = UNSET
        else:
            region = SandboxOptionsRegion(_region)

        setup = cast(list[str], d.pop("setup", UNSET))

        _snapshot = d.pop("snapshot", UNSET)
        snapshot: SandboxOptionsSnapshot | Unset
        if isinstance(_snapshot, Unset):
            snapshot = UNSET
        else:
            snapshot = SandboxOptionsSnapshot(_snapshot)

        timeout_ms = d.pop("timeout_ms", UNSET)

        sandbox_options = cls(
            allowed_domains=allowed_domains,
            cpu=cpu,
            disk_gb=disk_gb,
            image=image,
            memory_gb=memory_gb,
            network=network,
            network_allow_list=network_allow_list,
            region=region,
            setup=setup,
            snapshot=snapshot,
            timeout_ms=timeout_ms,
        )

        return sandbox_options
