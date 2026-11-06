from enum import StrEnum


class ErrorType(StrEnum):
    AUTHENTICATION = "authentication"
    CONFLICT = "conflict"
    GONE = "gone"
    INTERNAL = "internal"
    INVALID_REQUEST = "invalid_request"
    METHOD_NOT_ALLOWED = "method_not_allowed"
    NOT_ACCEPTABLE = "not_acceptable"
    NOT_FOUND = "not_found"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    PAYMENT_REQUIRED = "payment_required"
    PERMISSION = "permission"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def _decode(cls, value: object) -> "ErrorType":
        """Reads the value from an answer. A string this client does not list yet is kept as a member of its own,
        so an answer from a newer router still reads, and is written back as it came."""
        try:
            return cls(value)
        except ValueError:
            if not isinstance(value, str):
                raise
            member = str.__new__(cls, value)
            member._name_ = value
            member._value_ = value
            return member
