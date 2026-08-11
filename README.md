# bggpt-toolkit

Reliability and safety utilities for building products on [BgGPT](https://bggpt.ai) — INSAIT's
open Bulgarian LLM, served at `api.bggpt.ai` with an OpenAI-compatible API.

[Български](README.bg.md)

## Why this exists

BgGPT's API is fully OpenAI-compatible, so plain access needs no SDK — the standard `openai`
Python client already works, pointed at `https://api.bggpt.ai/v1`. What's missing is tooling
around a few reliability quirks specific to BgGPT itself, found live-testing it in production:

- **Vendor identity leaks.** Asked "what AI are you?", BgGPT answers with its full baked-in
  identity ("Аз съм BgGPT, създаден от INSAIT, базиран на Gemma-3...") — leaking the underlying
  model/vendor regardless of what your system prompt says your product's identity should be.
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
resp = c.chat.completions.create(model="bggpt-27b", messages=[{"role": "user", "content": "Здравей!"}])
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

```python
from bggpt_toolkit import IdentityGuard

guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="Аз съм „Моят асистент“ — ...",
    answer_en="I'm My Assistant — ...",
)

if identity_guard.is_identity_question(user_text):
    return guard.answer(user_text)  # short-circuit, no model call

# Otherwise, redact any incidental vendor mention in the model's answer:
clean_text, n = guard.redact(model_text)
```

For streaming answers, `guard.safe_flush_point(buffer, upto)` tells you how much of a growing
buffer is safe to flush now without risking a watched token (e.g. "BgGPT") arriving split across
two deltas ("bg" then "gpt").

### Tool-calling loop

```python
from bggpt_toolkit import run_tool_loop
from bggpt_toolkit.client import async_client

client = async_client()

async def execute_tool(name: str, arguments: dict) -> str:
    ...  # call your actual tool, return its result as a string

async for event in run_tool_loop(
    client,
    model="bggpt-27b",
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
