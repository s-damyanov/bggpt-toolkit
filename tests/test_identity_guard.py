"""Offline unit tests for IdentityGuard — pure logic, no network."""

from __future__ import annotations

import pytest

from bggpt_toolkit.identity_guard import IdentityGuard, is_identity_question


@pytest.fixture
def guard() -> IdentityGuard:
    return IdentityGuard(
        product_name="Test Assistant",
        answer_bg="Аз съм „Тестов асистент“.",
        answer_en="I'm Test Assistant.",
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


def test_redact_basic(guard: IdentityGuard) -> None:
    for leak in [
        "Аз съм BgGPT.", "I am BG-GPT", "разработен от INSAIT", "based on Gemma",
        "базиран на Gemma-3 27B", "Аз съм инсайт модел", "bggpt",
    ]:
        red, n = guard.redact(leak)
        assert n >= 1, leak
        for bad in ("bggpt", "bg-gpt", "insait", "инсайт", "gemma", "гема"):
            assert bad not in red.lower(), (leak, red)
    assert guard.product_name in guard.redact("Аз съм BgGPT")[0]


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


def test_redact_no_false_positives(guard: IdentityGuard) -> None:
    # substrings inside larger words must NOT be touched
    for ok in ["регистрация по ДДС", "гемоглобин е висок", "чл. 97а ЗДДС", "Договорът"]:
        red, n = guard.redact(ok)
        assert n == 0, (ok, red)
        assert red == ok


def test_redact_idempotent(guard: IdentityGuard) -> None:
    once, _ = guard.redact("Аз съм BgGPT, разработен от INSAIT.")
    twice, n = guard.redact(once)
    assert n == 0 and twice == once


def test_streaming_split_token(guard: IdentityGuard) -> None:
    # the exact hazard: a watched token split across deltas must not leak in halves
    assert "bggpt" not in _sim_stream(guard, ["Аз съм ", "bg", "gpt", ", помощник."]).lower()
    assert "insait" not in _sim_stream(guard, ["От ", "INS", "AIT", " съм."]).lower()
    assert "bg-gpt" not in _sim_stream(guard, ["", "bg", "-", "gpt", "!"]).lower()
    clean = ["Кратък ", "отговор: ", "прагът е ", "100 000 лв."]
    assert _sim_stream(guard, clean) == "".join(clean)


def test_streaming_no_stall(guard: IdentityGuard) -> None:
    got = _sim_stream(guard, ["Плащате ", "данъци", " и ", "осигуровки."])
    assert got == "Плащате данъци и осигуровки."


def test_extra_watched_forms() -> None:
    guard = IdentityGuard(
        product_name="X", answer_bg="x", answer_en="x",
        extra_watched_forms=("my-internal-codename",),
    )
    red, n = guard.redact("Built on my-internal-codename under the hood.")
    assert n == 1
    assert "my-internal-codename" not in red.lower()
    # defaults are still active alongside the extra form
    _, n2 = guard.redact("Powered by BgGPT.")
    assert n2 == 1
