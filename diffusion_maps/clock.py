"""Clock for keeping track of the wall time.

"""


__all__ = ['ClockError', 'Clock', 'log']

import datetime
import logging
import time

from typing import Optional     # noqa: F401. Used for mypy.


class ClockError(Exception):
    """Invalid clock operation."""
    pass


class Clock:
    """Clock for keeping track of time.

    """
    def __init__(self) -> None:
        self.start = None       # type: Optional[float]
        self.stop = None        # type: Optional[float]

    def tic(self) -> None:
        """Start the clock."""
        self.start = time.monotonic()
        self.stop = None

    def toc(self) -> None:
        """Stop the clock."""
        assert self.start is not None
        self.stop = time.monotonic()

    def __str__(self) -> str:
        """Human-readable representation of elapsed time."""
        if self.start is None:
            raise ClockError('The clock has not been started')
        else:
            start = datetime.datetime.fromtimestamp(self.start)

        if self.stop is None:
            stop = datetime.datetime.fromtimestamp(time.monotonic())
        else:
            stop = datetime.datetime.fromtimestamp(self.stop)

        delta = stop - start

        return str(delta)

    def __enter__(self):
        if self.start is None and self.stop is None:
            self.tic()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.start is not None:
            self.toc()


def log(function):
    """Create a decorator that logs the elapsed time.

    """
    def wrapper(*args, **kwargs):
        with Clock() as clock:
            result = function(*args, **kwargs)
            logging.debug('Completed {} after {} seconds.'
                          .format(function.__name__, clock))

        return result

    return wrapper
