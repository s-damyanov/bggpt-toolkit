"""Offline tests for RateLimiter — mostly a fake clock and no real sleeping, except the
concurrency regression test below, which needs real timing (see its docstring for why)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from bggpt_toolkit.ratelimit import RateLimiter


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_reserves_slot_when_free() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60.0, clock=clock)
    assert limiter._reserve_or_wait() == 0.0
    assert limiter._reserve_or_wait() == 0.0


def test_returns_wait_time_when_exhausted() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window=60.0, clock=clock)
    limiter._reserve_or_wait()
    limiter._reserve_or_wait()
    wait = limiter._reserve_or_wait()
    assert wait == pytest.approx(60.1)


def test_old_calls_fall_out_of_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window=60.0, clock=clock)
    limiter._reserve_or_wait()
    assert limiter._reserve_or_wait() > 0  # exhausted
    clock.advance(60.01)
    assert limiter._reserve_or_wait() == 0.0  # window has rolled past the first call


def test_acquire_sleeps_only_when_exhausted(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window=60.0, clock=clock)
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock.advance(s)  # retry loop re-checks the clock; a no-op sleep would spin forever

    monkeypatch.setattr("bggpt_toolkit.ratelimit.time.sleep", fake_sleep)

    limiter.acquire()
    assert sleeps == []  # first call: slot was free, no sleep

    limiter.acquire()
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(60.1)


async def test_acquire_async_sleeps_only_when_exhausted(monkeypatch) -> None:
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window=60.0, clock=clock)
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        clock.advance(s)  # retry loop re-checks the clock; a no-op sleep would spin forever

    monkeypatch.setattr("bggpt_toolkit.ratelimit.asyncio.sleep", fake_sleep)

    await limiter.acquire_async()
    assert sleeps == []

    await limiter.acquire_async()
    assert len(sleeps) == 1


async def test_async_waiters_do_not_over_admit() -> None:
    """Regression test for the thundering-herd bug: every blocked waiter woke up and appended a
    slot unconditionally, without rechecking whether the window still had room.

    This needs real time and real `asyncio.sleep`, not the fake clock used elsewhere in this file.
    A fake clock that advances synchronously inside each `sleep()` call lets suspended waiters
    resume one at a time, each seeing a clock already moved past the others — which happens to
    dodge the bug it's meant to catch, since the appends land spread across separate windows
    instead of piled into one. Real waiters actually block concurrently and resume within
    microseconds of each other, which is what the bug depends on. Kept to a 0.05s window so the
    whole run stays well under a second.
    """
    limiter = RateLimiter(limit=3, window=0.05)
    admitted: list[float] = []

    async def one() -> None:
        await limiter.acquire_async()
        admitted.append(time.monotonic())

    await asyncio.gather(*[one() for _ in range(9)])

    assert len(admitted) == 9
    for t in admitted:  # no 0.05s window ever holds more than 3 admissions
        assert len([u for u in admitted if t - 0.05 <= u <= t]) <= 3


def test_threaded_waiters_do_not_over_admit() -> None:
    """Threaded equivalent of the async thundering-herd regression above: 20 real threads racing
    `acquire()` against a limit of 5 reliably over-admitted before the lock was added (verified
    empirically), and runs in well under a second, so it's worth keeping alongside the async case
    rather than relying on the lock being "self-evidently right"."""
    limiter = RateLimiter(limit=5, window=0.05)
    admitted: list[float] = []
    lock = threading.Lock()

    def one() -> None:
        limiter.acquire()
        with lock:
            admitted.append(time.monotonic())

    threads = [threading.Thread(target=one) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(admitted) == 20
    for t in admitted:  # no 0.05s window ever holds more than 5 admissions
        assert len([u for u in admitted if t - 0.05 <= u <= t]) <= 5
