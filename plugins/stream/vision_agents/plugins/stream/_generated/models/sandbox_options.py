from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

from ..types import UNSET, Unset

T = TypeVar("T", bound="SandboxOptions")


@_attrs_define
class SandboxOptions:
    """How the sandbox is built and how long code may run in it. Only meaningful with a sandbox. Omit it for the provider's
    own Python sandbox and a 30 second run.

        Attributes:
            cpu (int | Unset): CPUs for the sandbox. Zero is the provider's default.
            disk_gb (int | Unset): Disk for the sandbox, in GiB. Zero is the provider's default.
            image (str | Unset): The container image to build on, which must have Python, such as python:3.13-slim-bookworm.
                Empty with anything else set is a slim Python 3.13 image.
            memory_gb (int | Unset): Memory for the sandbox, in GiB. Zero is the provider's default.
            setup (list[str] | Unset): Shell commands run once on top of the image when it is built, such as installing
                packages. The provider keeps the built image, so only the first sandbox from a given setup waits for it.
            timeout_ms (int | Unset): How long one run of code may take, at most 30 minutes. Zero is 30 seconds. A run is
                still bounded by the deadline of the skill it was written for.
    """

    cpu: int | Unset = UNSET
    disk_gb: int | Unset = UNSET
    image: str | Unset = UNSET
    memory_gb: int | Unset = UNSET
    setup: list[str] | Unset = UNSET
    timeout_ms: int | Unset = UNSET

    def to_dict(self) -> dict[str, Any]:
        cpu = self.cpu

        disk_gb = self.disk_gb

        image = self.image

        memory_gb = self.memory_gb

        setup: list[str] | Unset = UNSET
        if not isinstance(self.setup, Unset):
            setup = self.setup

        timeout_ms = self.timeout_ms

        field_dict: dict[str, Any] = {}

        field_dict.update({})
        if cpu is not UNSET:
            field_dict["cpu"] = cpu
        if disk_gb is not UNSET:
            field_dict["disk_gb"] = disk_gb
        if image is not UNSET:
            field_dict["image"] = image
        if memory_gb is not UNSET:
            field_dict["memory_gb"] = memory_gb
        if setup is not UNSET:
            field_dict["setup"] = setup
        if timeout_ms is not UNSET:
            field_dict["timeout_ms"] = timeout_ms

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        cpu = d.pop("cpu", UNSET)

        disk_gb = d.pop("disk_gb", UNSET)

        image = d.pop("image", UNSET)

        memory_gb = d.pop("memory_gb", UNSET)

        setup = cast(list[str], d.pop("setup", UNSET))

        timeout_ms = d.pop("timeout_ms", UNSET)

        sandbox_options = cls(
            cpu=cpu,
            disk_gb=disk_gb,
            image=image,
            memory_gb=memory_gb,
            setup=setup,
            timeout_ms=timeout_ms,
        )

        return sandbox_options
