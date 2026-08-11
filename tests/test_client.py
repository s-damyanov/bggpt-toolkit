"""Offline tests for client.py — no network calls; OpenAI/AsyncOpenAI construction doesn't hit
the network, it only sets attributes."""

from __future__ import annotations

import pytest
from openai import AsyncOpenAI, OpenAI

from bggpt_toolkit.client import BGGPT_BASE_URL, async_client, client, is_bggpt_model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("bggpt-gemma-3-27b-fp8", True),
        ("bggpt-27b", True),
        ("gpt-4o", False),
        ("gemma-3-27b", False),
        ("", False),
    ],
)
def test_is_bggpt_model(model: str, expected: bool) -> None:
    assert is_bggpt_model(model) is expected


def test_client_uses_bggpt_base_url_and_explicit_key() -> None:
    c = client(api_key="explicit-key")
    assert isinstance(c, OpenAI)
    assert str(c.base_url).rstrip("/") == BGGPT_BASE_URL.rstrip("/")
    assert c.api_key == "explicit-key"


def test_async_client_uses_bggpt_base_url_and_explicit_key() -> None:
    c = async_client(api_key="explicit-key")
    assert isinstance(c, AsyncOpenAI)
    assert str(c.base_url).rstrip("/") == BGGPT_BASE_URL.rstrip("/")
    assert c.api_key == "explicit-key"


def test_client_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BGGPT_API_KEY", "from-env")
    c = client()
    assert c.api_key == "from-env"


def test_explicit_key_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BGGPT_API_KEY", "from-env")
    c = client(api_key="explicit-key")
    assert c.api_key == "explicit-key"


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BGGPT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BGGPT_API_KEY"):
        client()
