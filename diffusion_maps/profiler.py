"""Context manager for running code under the Python profiler.

"""


__all__ = ['Profiler']

import cProfile as profile


class Profiler:
    """Run code under the profiler and report at the end."""
    def __init__(self, stream):
        """Initialize profiler with an open stream."""
        self.stream = stream

    def __enter__(self):
        if self.stream is not None:
            self.profile = profile.Profile()
            self.profile.enable()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream is not None:
            self.profile.disable()

            import pstats
            ps = pstats.Stats(self.profile, stream=self.stream)
            ps.sort_stats('cumulative')
            ps.print_stats()
