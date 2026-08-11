# Contributing

Issues and PRs welcome — in English or Bulgarian. This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Scope

This package stays narrow on purpose: reliability/safety fixes for BgGPT's own behavior, on top
of the standard `openai` client. Before adding a module, ask whether it's fixing something BgGPT
itself does (in scope) or something specific to one product's domain (out of scope — see
`docs/recipes/` for documenting a pattern without shipping the code).

`IdentityGuard` defaults to disclosure, not concealment — see its module docstring and the
README's licensing note for why (INSAIT's own ToS, `bggpt.ai/terms` Art. 5.8(2), plus the Gemma
Terms of Use it incorporates). PRs that flip that default, or that add new suppression behavior
enabled by default, won't be accepted.

[`.claude/skills/bggpt-toolkit/SKILL.md`](.claude/skills/bggpt-toolkit/SKILL.md) summarizes the
public API and known BgGPT quirks for coding agents. If a PR adds/removes/renames anything in the
public API (`bggpt_toolkit/__init__.py`'s `__all__`) or changes a default, update the skill file
in the same PR — it's meant to stay accurate, not aspirational.

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
