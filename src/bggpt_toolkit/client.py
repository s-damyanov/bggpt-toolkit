"""Thin, preconfigured `openai` client for BgGPT (api.bggpt.ai).

BgGPT's endpoint is fully OpenAI-Chat-Completions-compatible, so this module deliberately does
not reimplement API access — it only wires up the two things every integration needs: the base
URL and API key. Everything else (streaming, tool calls, retries) is the standard `openai` client.
"""

from __future__ import annotations

import os

from openai import AsyncOpenAI, OpenAI

BGGPT_BASE_URL = "https://api.bggpt.ai/v1"


def is_bggpt_model(model: str) -> bool:
    """True for BgGPT model names (e.g. "bggpt-gemma-3-27b-fp8"), as opposed to other engines a
    caller might be routing between (e.g. an OpenAI model)."""
    return model.startswith("bggpt-")


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("BGGPT_API_KEY")
    if not key:
        raise RuntimeError(
            "BgGPT API key not provided. Pass api_key=... or set the BGGPT_API_KEY env var."
        )
    return key


def client(api_key: str | None = None) -> OpenAI:
    """A sync `openai.OpenAI` client preconfigured for api.bggpt.ai."""
    return OpenAI(base_url=BGGPT_BASE_URL, api_key=_resolve_api_key(api_key))


def async_client(api_key: str | None = None) -> AsyncOpenAI:
    """An async `openai.AsyncOpenAI` client preconfigured for api.bggpt.ai."""
    return AsyncOpenAI(base_url=BGGPT_BASE_URL, api_key=_resolve_api_key(api_key))
