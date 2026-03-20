import sys
from typing import Callable, TypeVar

if sys.platform != "win32":
    import termios

    def safe_input(prompt: str) -> str:
        """Call input() after ensuring the terminal translates CR to NL.

        PortAudio (via sounddevice) can disable the ICRNL terminal flag,
        which causes Enter (CR) to show as ^M instead of submitting input.
        """
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            if not (attrs[0] & termios.ICRNL):
                attrs[0] |= termios.ICRNL
                termios.tcsetattr(fd, termios.TCSANOW, attrs)
        return input(prompt)

else:

    def safe_input(prompt: str) -> str:
        return input(prompt)


T = TypeVar("T")


def prompt_selection(
    items: list[T],
    formatter: Callable[[T], str],
    header: str,
    default: T | None = None,
    allow_skip: bool = False,
    empty_message: str | None = None,
) -> T | None:
    """Interactive terminal prompt to pick one item from a list."""
    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)

    if not items:
        if empty_message:
            print(f"  {empty_message}")
        print("-" * 50 + "\n")
        return None

    for i, item in enumerate(items):
        print(f"  {i}: {formatter(item)}")

    if allow_skip:
        print("  n: Skip (none)")

    print("-" * 50)

    while True:
        try:
            if allow_skip:
                text = f"Select [0-{len(items) - 1}] or 'n' to skip: "
            elif default is not None:
                text = f"Select [0-{len(items) - 1}] (Enter for default): "
            else:
                text = f"Select [0-{len(items) - 1}]: "

            choice = safe_input(text).strip().lower()

            if choice == "" and default is not None:
                print(f"  -> Using default: {formatter(default)}")
                return default

            if choice in ("n", "") and allow_skip:
                print("  -> No selection")
                print("-" * 50 + "\n")
                return None

            idx = int(choice)
            if 0 <= idx < len(items):
                selected = items[idx]
                print(f"  -> Selected: {formatter(selected)}")
                print("-" * 50 + "\n")
                return selected

            print(f"  Invalid choice, enter 0-{len(items) - 1}")
        except ValueError:
            print("  Please enter a number" + (" or 'n'" if allow_skip else ""))
