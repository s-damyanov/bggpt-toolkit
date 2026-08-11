# Contributing

Issues and PRs welcome — in English or Bulgarian.

## Scope

This package stays narrow on purpose: reliability/safety fixes for BgGPT's own behavior, on top
of the standard `openai` client. Before adding a module, ask whether it's fixing something BgGPT
itself does (in scope) or something specific to one product's domain (out of scope — see
`docs/recipes/` for documenting a pattern without shipping the code).

`IdentityGuard` defaults to disclosure, not concealment — see its module docstring and the
README's licensing note for why (Gemma Terms of Use / Prohibited Use Policy). PRs that flip that
default, or that add new suppression behavior enabled by default, won't be accepted.

If you've found another live BgGPT reliability quirk, please include:

- A minimal repro (prompt + observed output).
- Which model (`bggpt-*`) and roughly when you observed it — BgGPT's behavior can drift between
  versions, so dated findings are more useful than undated ones.

## Setup

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

`examples/live_smoke.py` needs a real `BGGPT_API_KEY` and is not part of CI — run it manually
before submitting a change to `client.py`, `ratelimit.py`, or `identity_guard.py`'s defaults, to
confirm the fix still matches BgGPT's current live behavior.
