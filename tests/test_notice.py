"""Offline tests for notice.render() — pure string templating, no network."""

from __future__ import annotations

import pytest

from bggpt_toolkit.notice import (
    BGGPT_TERMS_URL,
    GEMMA_PROHIBITED_USE_URL,
    GEMMA_TERMS_URL,
    INSAIT_URL,
    render,
)


def test_default_lang_is_bulgarian() -> None:
    assert render("Моят асистент") == render("Моят асистент", lang="bg")


@pytest.mark.parametrize("lang", ["bg", "en"])
def test_contains_product_name(lang: str) -> None:
    text = render("Test Product", lang=lang)
    assert "Test Product" in text


@pytest.mark.parametrize("lang", ["bg", "en"])
def test_contains_required_disclosures(lang: str) -> None:
    text = render("Test Product", lang=lang)
    assert "BgGPT" in text
    assert "INSAIT" in text
    assert "Gemma" in text
    for url in (INSAIT_URL, BGGPT_TERMS_URL, GEMMA_TERMS_URL, GEMMA_PROHIBITED_USE_URL):
        assert url in text


def test_invalid_lang_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported lang"):
        render("Test Product", lang="fr")


def test_product_name_can_contain_braces_safely() -> None:
    # product names shouldn't be interpreted as further format fields
    text = render("Product {with} braces", lang="en")
    assert "Product {with} braces" in text
