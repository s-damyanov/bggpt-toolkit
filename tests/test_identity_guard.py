"""Offline unit tests for IdentityGuard — pure logic, no network."""

from __future__ import annotations

import pytest

from bggpt_toolkit.identity_guard import IdentityGuard, is_identity_question


@pytest.fixture
def guard() -> IdentityGuard:
    """Default configuration: disclose (suppress_incidental_mentions=False)."""
    return IdentityGuard(
        product_name="Test Assistant",
        answer_bg="Аз съм „Тестов асистент“, базиран на BgGPT (INSAIT).",
        answer_en="I'm Test Assistant, built on BgGPT (INSAIT).",
    )


@pytest.fixture
def suppressing_guard() -> IdentityGuard:
    """Opt-in configuration: suppress incidental vendor mentions."""
    return IdentityGuard(
        product_name="Test Assistant",
        answer_bg="Аз съм „Тестов асистент“.",
        answer_en="I'm Test Assistant.",
        suppress_incidental_mentions=True,
    )


def _sim_stream(guard: IdentityGuard, chunks: list[str]) -> str:
    """Reproduce a streaming consumer's flush logic (hold-back + redaction) over deltas."""
    buffer, flushed, out = "", 0, []
    for delta in chunks:
        buffer += delta
        safe = guard.safe_flush_point(buffer, len(buffer))
        if safe > flushed:
            piece, _ = guard.redact(buffer[flushed:safe])
            out.append(piece)
            flushed = safe
    if flushed < len(buffer):
        piece, _ = guard.redact(buffer[flushed:])
        out.append(piece)
    return "".join(out)


def test_default_does_not_redact(guard: IdentityGuard) -> None:
    # disclosure is the default: incidental vendor mentions pass through unchanged
    for text in ["Аз съм BgGPT.", "based on Gemma", "разработен от INSAIT"]:
        red, n = guard.redact(text)
        assert n == 0
        assert red == text


def test_default_streaming_is_pass_through(guard: IdentityGuard) -> None:
    chunks = ["Аз съм ", "bg", "gpt", ", помощник."]
    assert _sim_stream(guard, chunks) == "".join(chunks)


def test_suppress_redacts_vendor_mentions(suppressing_guard: IdentityGuard) -> None:
    for mention in [
        "Аз съм BgGPT.", "I am BG-GPT", "разработен от INSAIT", "based on Gemma",
        "базиран на Gemma-3 27B", "Аз съм инсайт модел", "bggpt",
    ]:
        red, n = suppressing_guard.redact(mention)
        assert n >= 1, mention
        for bad in ("bggpt", "bg-gpt", "insait", "инсайт", "gemma", "гема"):
            assert bad not in red.lower(), (mention, red)
    assert suppressing_guard.product_name in suppressing_guard.redact("Аз съм BgGPT")[0]


def test_identity_question_detected() -> None:
    for q in [
        "Какъв изкуствен интелект си?", "Кой те е разработил?", "What AI are you?",
        "Who made you?", "На какъв модел си базиран?", "Как се казваш?",
    ]:
        assert is_identity_question(q), q


def test_answer_matches_language(guard: IdentityGuard) -> None:
    assert "Тестов асистент" in guard.answer("Какъв AI си?")
    assert guard.answer("What AI are you?").startswith("I'm")


def test_identity_question_no_false_positive() -> None:
    for q in [
        "Какъв модел на облагане важи за ЕООД?", "Как се регистрира компания в НАП?",
        "Какъв данък дължа?",
    ]:
        assert not is_identity_question(q), q


def test_suppress_no_false_positives(suppressing_guard: IdentityGuard) -> None:
    # substrings inside larger words must NOT be touched
    for ok in ["регистрация по ДДС", "гемоглобин е висок", "чл. 97а ЗДДС", "Договорът"]:
        red, n = suppressing_guard.redact(ok)
        assert n == 0, (ok, red)
        assert red == ok


def test_suppress_idempotent(suppressing_guard: IdentityGuard) -> None:
    once, _ = suppressing_guard.redact("Аз съм BgGPT, разработен от INSAIT.")
    twice, n = suppressing_guard.redact(once)
    assert n == 0 and twice == once


def test_suppress_streaming_split_token(suppressing_guard: IdentityGuard) -> None:
    # the exact hazard: a watched token split across deltas must not survive in halves
    assert "bggpt" not in _sim_stream(
        suppressing_guard, ["Аз съм ", "bg", "gpt", ", помощник."]
    ).lower()
    assert "insait" not in _sim_stream(suppressing_guard, ["От ", "INS", "AIT", " съм."]).lower()
    assert "bg-gpt" not in _sim_stream(suppressing_guard, ["", "bg", "-", "gpt", "!"]).lower()
    clean = ["Кратък ", "отговор: ", "прагът е ", "100 000 лв."]
    assert _sim_stream(suppressing_guard, clean) == "".join(clean)


def test_suppress_streaming_no_stall(suppressing_guard: IdentityGuard) -> None:
    got = _sim_stream(suppressing_guard, ["Плащате ", "данъци", " и ", "осигуровки."])
    assert got == "Плащате данъци и осигуровки."


def test_extra_watched_forms_only_apply_when_suppressing() -> None:
    guard = IdentityGuard(
        product_name="X", answer_bg="x", answer_en="x",
        extra_watched_forms=("my-internal-codename",),
        suppress_incidental_mentions=True,
    )
    red, n = guard.redact("Built on my-internal-codename under the hood.")
    assert n == 1
    assert "my-internal-codename" not in red.lower()
    # defaults are still active alongside the extra form
    _, n2 = guard.redact("Powered by BgGPT.")
    assert n2 == 1
