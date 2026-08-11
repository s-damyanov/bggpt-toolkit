"""Offline tests for RateLimiter — a fake clock, no real sleeping."""

from __future__ import annotations

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
    monkeypatch.setattr("bggpt_toolkit.ratelimit.time.sleep", lambda s: sleeps.append(s))

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

    monkeypatch.setattr("bggpt_toolkit.ratelimit.asyncio.sleep", fake_sleep)

    await limiter.acquire_async()
    assert sleeps == []

    await limiter.acquire_async()
    assert len(sleeps) == 1
