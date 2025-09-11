from __future__ import annotations

from collections.abc import Callable

from typing import Any, Concatenate, ParamSpec, TypeVar
import functools

# ---------- The function whose signature we want to reuse ----------
def _echo(
    text: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> str:
    return text

# ---------- Typing setup ----------
P = ParamSpec("P")     # will be bound to _echo's parameters
R = TypeVar("R")       # will be bound to _echo's return type
T = TypeVar("T", bound="MyClass")  # the instance type (self)

# ---------- The decorator factory ----------
def wrap_echo(target: Callable[P, R]) -> Callable[
    [Callable[Concatenate[T, P], R]],
    Callable[Concatenate[T, P], R]
]:
    def decorator(method: Callable[Concatenate[T, P], R]) -> Callable[Concatenate[T, P], R]:
        @functools.wraps(method)
        def wrapper(self: T, *args: P.args, **kwargs: P.kwargs) -> R:
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

# ---------- Usage on an instance method ----------
class MyClass:
    @wrap_echo(_echo)
    def class_create(self, *args: P.args, **kwargs: P.kwargs) -> R:
        # The body is not used because the decorator replaces it,
        # but keeping the signature here lets IDEs show proper hints even before decoration.
        return _echo(*args, **kwargs)

# ---------- Example calls (with full typing support propagated) ----------
mc = MyClass()

result = mc.class_create(
    mc,
    text="hi",
    system="assistant",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens="42",
)