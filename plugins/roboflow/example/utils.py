import time


class Debouncer:
    """
    A simple object to debounce repeated function calls.
    """

    def __init__(self, interval: float):
        self.last_called = 0.0
        self.interval = interval

    def __bool__(self):
        now = time.monotonic()
        if now - self.last_called > self.interval:
            self.last_called = now
            return True
        else:
            return False
