from enum import StrEnum


class SessionEndFault(StrEnum):
    CALLER = "caller"
    CARRIER = "carrier"
    CUSTOMER_CONFIG = "customer_config"
    CUSTOMER_CREDIT = "customer_credit"
    NONE = "none"
    ROUTER = "router"
    VENDOR = "vendor"

    def __str__(self) -> str:
        return str(self.value)
