# bggpt-toolkit

[![CI](https://github.com/s-damyanov/bggpt-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/s-damyanov/bggpt-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Reliability and safety utilities for building products on [BgGPT](https://bggpt.ai) — INSAIT's
open Bulgarian LLM, served at `api.bggpt.ai` with an OpenAI-compatible API.

[Български](README.bg.md)

## Why this exists

BgGPT's API is fully OpenAI-compatible, so plain access needs no SDK — the standard `openai`
Python client already works, pointed at `https://api.bggpt.ai/v1`. What's missing is tooling
around a few reliability quirks specific to BgGPT itself, found live-testing it in production:

- **Persona-override inconsistency.** Even with a system prompt asking for a custom persona,
  asked "what AI are you?" BgGPT sometimes answers with its own baked-in identity instead ("Аз съм
  BgGPT, създаден от INSAIT, базиран на Gemma-3..."). This isn't a security flaw — BgGPT is
  INSAIT's own openly attributed model, and INSAIT plainly wants it known as BgGPT — it's just
  inconsistent about which identity it adopts. See [`identity_guard.py`](src/bggpt_toolkit/identity_guard.py)
  and the [licensing note](#licensing-note) below for why this package's default is to make that
  disclosure *consistent*, not to help you conceal it.
- **Fabricated tool calls.** Given a question that clearly needs a tool, under soft prompting
  BgGPT can narrate a *fake* call as literal answer text (`"[get_status()]"`) instead of emitting
  a real structured `tool_calls` delta.
- **Hard rate limit.** `api.bggpt.ai` enforces 20 requests/minute per API key, server-side.

These aren't business logic — they're BgGPT's own behavior, independent of what you're building
on top of it — so they belong in a shared package rather than re-solved per project. This one is
deliberately narrow: a thin client wrapper plus the fixes above, not a reimplementation of API
access.

## Install

```bash
pip install git+https://github.com/s-damyanov/bggpt-toolkit.git
```

(PyPI release planned once this stabilizes.)

## Usage

### Client

```python
from bggpt_toolkit import client

c = client()  # reads BGGPT_API_KEY from the environment, or pass api_key=...
resp = c.chat.completions.create(model="bggpt-gemma-3-27b-fp8", messages=[{"role": "user", "content": "Здравей!"}])
```

An `async_client()` counterpart returns an `AsyncOpenAI`.

### Rate limiting

```python
from bggpt_toolkit import RateLimiter

limiter = RateLimiter(limit=20, window=60.0)  # matches api.bggpt.ai's stated limit
limiter.acquire()          # sync: blocks until a slot is free
await limiter.acquire_async()  # async: does not block the event loop
```

A ready-made instance at BgGPT's stated limit is available as
`bggpt_toolkit.ratelimit.bggpt_rate_limiter`.

### Identity guard

Default behavior is **disclosure, not concealment** — see [licensing note](#licensing-note) below
for why.

```python
from bggpt_toolkit import IdentityGuard

guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="Аз съм „Моят асистент“ — базиран на BgGPT (INSAIT).",
    answer_en="I'm My Assistant — built on BgGPT (INSAIT).",
    own_names=("Моят асистент",),  # every spelling the product calls itself
)

if guard.is_identity_question(user_text):
    return guard.answer(user_text)      # cheap pre-filter: skips the API call entirely

text = call_bggpt(...)
text, replaced = guard.enforce_answer(user_text, text)   # the part that actually guarantees it
```

Use both halves. Detecting the *question* is the weak side of this problem — there is no closed
set of ways to ask "what are you", in either language — so `is_identity_question` can only
approximate it. Detecting the *claim* is the strong side, and `enforce_answer` is built as a
**closed-world allow-list, not a blocklist**: rather than scanning for known-bad vendor names, it
extracts whatever name an answer actually claims and checks it against what your product is
allowed to claim (`own_names`, plus `may_disclose` for an honest "built on BgGPT" attribution). A
blocklist can only ever catch identities someone thought to enumerate in advance — it cannot catch
a model released after the list was written, or a wholly invented name like "Асистент-Про 3000,
разработен от Балкан Софт", which names no real vendor at all. The allow-list catches both,
because neither is in the list of names *this* product is allowed to claim, whatever they are.

That matters because a missed question is exactly when BgGPT improvises — including, observed
live at `temperature=0`, a confident fabrication that names a vendor with no connection to it at
all ("Аз съм GPT-3.5 на OpenAI"). `enforce_answer` replaces any disallowed claim with your own
`answer()`; a correct disclosure (naming only `own_names`/`may_disclose`) passes through
unchanged, so re-running this is a no-op. It needs whole sentences, so a streaming caller should
hold back the first sentence until it can be checked — in practice identity claims land at the
very start.

`may_disclose` defaults to `identity_guard.DEFAULT_MAY_DISCLOSE` — BgGPT's own name and vendor,
the truthful attribution chain Art. 5.8(2) requires. It deliberately excludes Gemma: BgGPT
genuinely is Gemma-derived, but a product that wants to volunteer that in-chat should say so
explicitly —

```python
guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="...", answer_en="...",
    may_disclose=identity_guard.DEFAULT_MAY_DISCLOSE + ("Gemma",),
)
```

For debugging or logging, `identity_guard.identity_claims(text)` returns every claim found
(allowed or not) as `IdentityClaim(kind, name, sentence)` — `contains_identity_claim`/
`enforce_answer` are thin wrappers that apply the allow-list on top of it.

`is_identity_question` is a fast, dependency-free pattern match — it has an inherent recall
ceiling, so if your product already computes embeddings for retrieval, you can plug in a semantic
fallback for phrasings the pattern list misses. It's only consulted when the fast path misses,
and it only ever affects latency (a wasted API call), never correctness, now that `enforce_answer`
exists — so it's fine to leave low-recall if you don't have embeddings to hang a fallback off of.
See [`docs/recipes/identity-prefilter.md`](docs/recipes/identity-prefilter.md) for better options
than growing the phrase list by hand, including reusing a scope gate you may already have:

```python
guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="...", answer_en="...",
    similarity_fn=lambda text: my_cosine_similarity(text, my_identity_question_embeddings),
    similarity_threshold=0.55,  # calibrate against your own examples, same as docs/recipes/scope-gate.md
)
```

See `identity_guard.IDENTITY_QUESTION_EXEMPLARS` for a starting set of canonical identity
questions to embed and compare against.

`guard.redact(text)` is a no-op by default. If your product has made its own informed decision to
suppress incidental vendor mentions mid-answer, opt in explicitly:

```python
guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="...", answer_en="...",
    suppress_incidental_mentions=True,
)
clean_text, n = guard.redact(model_text)
```

For streaming answers with suppression enabled, `guard.safe_flush_point(buffer, upto)` tells you
how much of a growing buffer is safe to flush now without risking a watched token (e.g. "BgGPT")
arriving split across two deltas ("bg" then "gpt"). It's a no-op too when suppression is off.

### End-user notice

`IdentityGuard` only covers what happens if a user directly asks "what AI are you?" — most never
do, so it likely isn't sufficient by itself for the [Art. 5.8(2) notice requirement](#licensing-note)
below. `notice.render()` returns ready-to-adapt text for a static notice (footer, about page,
rights section) instead:

```python
from bggpt_toolkit import render_notice

text_bg = render_notice("My Assistant")            # lang="bg" is the default
text_en = render_notice("My Assistant", lang="en")
```

Adapt the wording to your product — see the module's docstring for what it does and doesn't cover.

### Tool-calling loop

```python
from bggpt_toolkit import run_tool_loop
from bggpt_toolkit.client import async_client

client = async_client()

async def execute_tool(name: str, arguments: dict) -> str:
    ...  # call your actual tool, return its result as a string

async for event in run_tool_loop(
    client,
    model="bggpt-gemma-3-27b-fp8",
    messages=messages,           # mutated in place with each turn
    tools=my_chat_tools,
    execute_tool=execute_tool,
    max_rounds=3,
    rate_limit=limiter.acquire_async,
):
    if event["type"] == "delta":
        print(event["text"], end="")
```

`run_tool_loop` forces a real structured tool call on the first round (instead of letting BgGPT
narrate a fake one as text) and always appends one tools-omitted final round, so a model that
keeps calling tools every round still can't leave you with total silence.

## For AI coding agents

[`.claude/skills/bggpt-toolkit/SKILL.md`](.claude/skills/bggpt-toolkit/SKILL.md) packages the same
hard-won knowledge above (which modules to reach for, BgGPT's non-deterministic identity-leak
behavior, the Art. 5.8(2) notice requirement) as a Claude Code Agent Skill, so a coding agent
working in your project reaches for this library correctly instead of reimplementing its fixes
from scratch.

It isn't discovered automatically just by installing this package — copy the
`.claude/skills/bggpt-toolkit/` directory into your own project's `.claude/skills/` to use it.

## What's *not* in here

An out-of-scope/off-topic gate (deciding whether a question is even worth sending to the model)
is a genuinely useful pattern, but its keyword lists are inherently per-product — there's nothing
generic to ship. See [`docs/recipes/scope-gate.md`](docs/recipes/scope-gate.md) for the pattern.

## Licensing note

Two sources of terms apply to anyone building on `api.bggpt.ai`, both worth reading directly:

- **INSAIT's own [Terms of Service](https://bggpt.ai/terms)** for the API. Art. 5.8(2) requires
  that you "explicitly notify End Users that the applications/services/products they access are
  based on the BgGPT Model" — a stated term of using the API, not an inference. (Art. 5.8(8)(a)
  also prohibits bulk/automated scraping and overload requests — one more reason to always run
  requests through `ratelimit.py` rather than skip it.)
- Accepting those terms also binds you to Google's **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)**
  (Art. 1.7), whose [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
  separately restricts misleading claims of expertise or capability in sensitive areas — health,
  finance, government services, legal — and bars unlicensed practice of professions like legal,
  financial, or medical/health advice.

The disclosure obligation is about your *product* giving end users notice somewhere — a footer, an
about page, a rights/ToS section — not necessarily that the chatbot itself must volunteer it in
every reply. `IdentityGuard.answer()` covers the direct-question case; `render_notice()` (see
[usage](#end-user-notice) above) gives you ready-to-adapt text for the static notice, which is
likely the part that actually satisfies Art. 5.8(2) since most users never ask directly. Either
way, concealment isn't a compliant starting point, which is why this package defaults to
disclosure; suppression (`suppress_incidental_mentions=True`) is opt-in and is your call to make
for your own product, not something this package recommends.

This is not legal advice — read the actual terms above if this applies to your use case.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

`examples/live_smoke.py` is an opt-in script that hits the real `api.bggpt.ai` (requires
`BGGPT_API_KEY`); it's not part of the test suite or CI.

## License

MIT — see [LICENSE](LICENSE).
