from enum import StrEnum


class SipTrunkTransport(StrEnum):
    TCP = "tcp"
    TLS = "tls"
    UDP = "udp"

    def __str__(self) -> str:
        return str(self.value)
