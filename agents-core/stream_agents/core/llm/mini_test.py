from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def call_twice(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, R]:
    """Calls the given function twice with the same arguments and returns both results."""
    first = func(*args, **kwargs)
    second = func(*args, **kwargs)
    return first, second


# Example usage
def greet(name: str, excited: bool = False) -> str:
    return f"Hello {name}{'!!!' if excited else ''}"


result = call_twice(greet, "Thierry", excited=True)

call_twice(greet, "hi", excited=True)
