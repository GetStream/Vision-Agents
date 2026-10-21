from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from typing_extensions import Self

T = TypeVar("T", bound="ConnectorSandbox")


@_attrs_define
class ConnectorSandbox:
    """How code in an agent's sandbox reaches the provider's API with a connection's credential.

    Attributes:
        env (str): The env var code in the sandbox reads, such as GITHUB_TOKEN. Its value is a placeholder: the
            sandbox's egress puts the connection's credential in its place, only in a request header and only to one of
            hosts.
        hosts (list[str]): The hosts a request may carry the credential to: an exact host, or *. and a domain for the
            domain and every name under it.
    """

    env: str
    hosts: list[str]

    def to_dict(self) -> dict[str, Any]:
        env = self.env

        hosts = self.hosts

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "env": env,
                "hosts": hosts,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls, src_dict: Mapping[str, Any]) -> Self:
        d = dict(src_dict)
        env = d.pop("env")

        hosts = cast(list[str], d.pop("hosts"))

        connector_sandbox = cls(
            env=env,
            hosts=hosts,
        )

        return connector_sandbox
