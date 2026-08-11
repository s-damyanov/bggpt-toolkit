# bggpt-toolkit

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
)

if identity_guard.is_identity_question(user_text):
    return guard.answer(user_text)  # a consistent, honest answer, instead of leaving it to chance
```

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

## What's *not* in here

An out-of-scope/off-topic gate (deciding whether a question is even worth sending to the model)
is a genuinely useful pattern, but its keyword lists are inherently per-product — there's nothing
generic to ship. See [`docs/recipes/scope-gate.md`](docs/recipes/scope-gate.md) for the pattern.

## Licensing note

Two source of terms apply to anyone building on `api.bggpt.ai`, both worth reading directly:

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
