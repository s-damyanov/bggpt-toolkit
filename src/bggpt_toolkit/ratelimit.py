"""Client-side throttle for api.bggpt.ai's server-enforced rate limit.

INSAIT states a 20 requests/minute limit per API key on api.bggpt.ai. It's easy to blow past that
in anything that makes more than one BgGPT call per turn (e.g. a query-planning call followed by
an answer call, or a batch eval run) without client-side pacing — the alternative is discovering
the limit via 429s. This is a plain sliding-window limiter: shared across every call site that
uses the same API key, since the limit is enforced server-side per key, not per code path.
"""

from __future__ import annotations

import asyncio
import threading
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
        # Guards `_call_times`. Deliberately a threading.Lock, not an asyncio.Lock: this class
        # exports a process-wide singleton (`bggpt_rate_limiter`) that can legitimately be shared
        # across more than one event loop (e.g. a sync script and an async one both importing the
        # module), and an asyncio.Lock bound to whichever loop constructs it would deadlock or
        # raise if awaited from another. A threading.Lock works from both sync and async callers as
        # long as it is never held across an `await`/sleep, which it isn't here.
        self._lock = threading.Lock()

    def _reserve_or_wait(self) -> float:
        """0.0 if a slot was reserved; otherwise seconds to wait before RETRYING.

        The non-zero return reserves nothing — callers must sleep and call this again, rather than
        appending a slot themselves after sleeping. Every blocked caller wakes at roughly the same
        time, so a caller that appends unconditionally admits the whole waiting set at once
        (measured: 20 callers through a limit of 5), which is exactly the burst this limiter exists
        to keep off api.bggpt.ai.
        """
        now = self._clock()
        while self._call_times and now - self._call_times[0] > self.window:
            self._call_times.popleft()
        if len(self._call_times) < self.limit:
            self._call_times.append(now)
            return 0.0
        return self.window - (now - self._call_times[0]) + 0.1

    def acquire(self) -> None:
        """Block (sync) until a request slot is free."""
        while True:
            with self._lock:
                wait = self._reserve_or_wait()
            if wait <= 0.0:
                return
            time.sleep(wait)

    async def acquire_async(self) -> None:
        """Async equivalent — must not block the event loop.

        The lock is only ever held for the duration of `_reserve_or_wait()`, never across the
        `await asyncio.sleep(wait)` below — so it's safe for a plain threading.Lock to guard state
        shared with the sync `acquire()` path too.
        """
        while True:
            with self._lock:
                wait = self._reserve_or_wait()
            if wait <= 0.0:
                return
            await asyncio.sleep(wait)


# Preconfigured at INSAIT's stated api.bggpt.ai limit, for convenience — share this instance
# across call sites using the same API key rather than constructing a new one per call.
bggpt_rate_limiter = RateLimiter(limit=20, window=60.0)
