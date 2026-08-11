"""Opt-in live smoke test against the real api.bggpt.ai.

Not run in CI and not part of the pytest suite — requires a real BGGPT_API_KEY and makes real
network calls. Run manually before releasing a change to client.py, ratelimit.py, or
identity_guard.py's defaults, since BgGPT's behavior can drift between model versions and these
modules encode fixes for specific observed behavior.

Usage:
    BGGPT_API_KEY=... python examples/live_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

from bggpt_toolkit import IdentityGuard, RateLimiter, client, is_bggpt_model
from bggpt_toolkit.identity_guard import is_identity_question

MODEL = os.getenv("BGGPT_MODEL", "bggpt-gemma-3-27b-fp8")


def check(label: str, condition: bool) -> None:
    status = "OK  " if condition else "FAIL"
    print(f"  {status} {label}")
    if not condition:
        global failed
        failed = True


failed = False


def main() -> None:
    if not os.getenv("BGGPT_API_KEY"):
        print("BGGPT_API_KEY not set — nothing to do.")
        return

    c = client()

    print(f"1. Basic call ({MODEL})")
    resp = c.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Кажи 'здравей' и нищо друго."}],
        temperature=0,
    )
    text = resp.choices[0].message.content or ""
    check("got non-empty response", bool(text.strip()))
    check("is_bggpt_model recognizes the model", is_bggpt_model(MODEL))
    print(f"       -> {text.strip()[:80]!r}")

    print("\n2. Rate limiter throttles a burst instead of hitting a 429")
    limiter = RateLimiter(limit=3, window=10.0)
    start = time.monotonic()
    for i in range(4):
        limiter.acquire()
        c.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": f"ping {i}"}], temperature=0,
            max_tokens=5,
        )
    elapsed = time.monotonic() - start
    check(f"4th call under limit=3/10s waited (elapsed={elapsed:.1f}s)", elapsed >= 9.5)

    print("\n3. Identity guard against a live identity question")
    # suppress_incidental_mentions=True here only to exercise the opt-in backstop mechanism end
    # to end — the library default is to disclose (see identity_guard.py's module docstring for
    # why: BgGPT's own vendor disclosure isn't a secret, and this package's default policy is
    # honesty, not concealment).
    guard = IdentityGuard(
        product_name="Smoke Test Assistant",
        answer_bg="Аз съм Smoke Test Assistant, базиран на BgGPT (INSAIT).",
        answer_en="I'm Smoke Test Assistant, built on BgGPT (INSAIT).",
        suppress_incidental_mentions=True,
    )
    question = "Какъв AI модел си и кой те е разработил?"
    check("is_identity_question flags the live probe question", is_identity_question(question))

    # BgGPT's disclosure phrasing is NOT deterministic even at temperature=0 — observed samples
    # include "разработен от INSAIT ... Gemma" (caught by the watched vocabulary) as well as a
    # vaguer "developed by Google" (not caught by design: "google" alone is far too generic a
    # token to safely blanket-redact, see identity_guard.py's docstring — is_identity_question()'s
    # short-circuit, not redact(), is the primary tool for the direct-question case; this loop
    # samples for the specific, narrow vocabulary redact() targets). Sample a few phrasings and
    # pass if the watched vocabulary shows up in at least one — a single sample is too noisy to
    # trust either way.
    prompts = [
        question,
        "На какъв модел (vendor/base model) си базиран? Кажи направо.",
        "What AI model are you built on? Answer plainly, name the vendor.",
    ]
    any_caught = False
    for q in prompts:
        resp = c.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are Smoke Test Assistant, a helpful assistant."},
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        _, n = guard.redact(raw)
        print(f"       [{q[:40]!r}] raw: {raw.strip()[:90]!r}  (n={n})")
        any_caught = any_caught or n > 0
    check(
        "current BgGPT disclosure phrasing is still caught by the default watched terms "
        "on at least one of several phrasings",
        any_caught,
    )

    print()
    if failed:
        print("Some checks FAILED — BgGPT's live behavior may have drifted from what these "
              "modules were built against. Investigate before releasing.")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
