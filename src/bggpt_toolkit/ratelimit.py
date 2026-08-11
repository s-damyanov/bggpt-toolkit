"""Client-side throttle for api.bggpt.ai's server-enforced rate limit.

INSAIT states a 20 requests/minute limit per API key on api.bggpt.ai. It's easy to blow past that
in anything that makes more than one BgGPT call per turn (e.g. a query-planning call followed by
an answer call, or a batch eval run) without client-side pacing — the alternative is discovering
the limit via 429s. This is a plain sliding-window limiter: shared across every call site that
uses the same API key, since the limit is enforced server-side per key, not per code path.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Sliding-window rate limiter. Call `acquire()` (or `await acquire_async()`) immediately
    before each request; it blocks until a slot is free."""

    def __init__(
        self,
        limit: int = 20,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window
        self._clock = clock
        self._call_times: deque[float] = deque()

    def _reserve_or_wait(self) -> float:
        """0.0 and reserves a slot if one is free now; otherwise returns seconds to wait
        (caller must sleep, then append the actual call time itself)."""
        now = self._clock()
        while self._call_times and now - self._call_times[0] > self.window:
            self._call_times.popleft()
        if len(self._call_times) < self.limit:
            self._call_times.append(now)
            return 0.0
        return self.window - (now - self._call_times[0]) + 0.1

    def acquire(self) -> None:
        """Block (sync) until a request slot is free."""
        wait = self._reserve_or_wait()
        if wait > 0:
            time.sleep(wait)
            self._call_times.append(self._clock())

    async def acquire_async(self) -> None:
        """Async equivalent — must not block the event loop."""
        wait = self._reserve_or_wait()
        if wait > 0:
            await asyncio.sleep(wait)
            self._call_times.append(self._clock())


# Preconfigured at INSAIT's stated api.bggpt.ai limit, for convenience — share this instance
# across call sites using the same API key rather than constructing a new one per call.
bggpt_rate_limiter = RateLimiter(limit=20, window=60.0)
