"""A clock that AppWorld can't freeze.

AppWorld uses freezegun to pin the date/time to each task's simulated "now". freezegun replaces
time.time / time.monotonic / time.perf_counter everywhere, including references other modules
hold at module level. We keep the real function inside a closure, which freezegun doesn't
touch, so latency measurements, wall time and rate-limit throttling keep working.
"""

from __future__ import annotations

import time


def _make_clock():
    real = time.perf_counter

    def now() -> float:
        return real()

    return now


now = _make_clock()
