"""Offline unit tests for IdentityGuard — pure logic, no network."""

from __future__ import annotations

import pytest

from bggpt_toolkit.identity_guard import (
    DEFAULT_MAY_DISCLOSE,
    IdentityGuard,
    contains_identity_claim,
    identity_claims,
    is_identity_question,
)


@pytest.fixture
def guard() -> IdentityGuard:
    """Default configuration: disclose (suppress_incidental_mentions=False)."""
    return IdentityGuard(
        product_name="Test Assistant",
        answer_bg="Аз съм „Тестов асистент“, базиран на BgGPT (INSAIT).",
        answer_en="I'm Test Assistant, built on BgGPT (INSAIT).",
        own_names=("Тестов асистент",),
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


def test_identity_question_detected_live_false_negatives() -> None:
    # Found live against a real deployment: each of these fell through the old pattern list for a
    # different reason (missing grammatical gender, "which" vs "what" phrasing, a second-person
    # verb variant other than "си") — see identity_guard.py's _IDENTITY_Q comment.
    for q in ["Какво си ти?", "Кой AI си ти?", "Какъв модел използваш?"]:
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


def test_which_pronoun_normalization_generalizes() -> None:
    # None of these exact gender/number forms is a literal entry anywhere in _IDENTITY_Q — they
    # pass purely because _normalize_which collapses какъв/каква/какво/какви and кой/коя/кое/кои
    # onto one canonical form each before matching.
    for q in [
        "Какви ai си?", "Коя AI си?", "Какви представляваш?", "Каква ai си?",
        "Кое компания те създаде?",
    ]:
        assert is_identity_question(q), q


def test_similarity_fn_used_only_as_fallback() -> None:
    calls: list[str] = []

    def similarity_fn(text: str) -> float:
        calls.append(text)
        return 0.9

    # fast-path hit: similarity_fn must not even be called
    assert is_identity_question("Какъв AI си?", similarity_fn=similarity_fn) is True
    assert calls == []

    # fast-path miss, similarity_fn above threshold -> True
    assert is_identity_question(
        "С какво разговарям в момента?", similarity_fn=similarity_fn
    ) is True
    assert calls == ["С какво разговарям в момента?"]


def test_similarity_fn_respects_threshold() -> None:
    miss_text = "С какво разговарям в момента?"
    assert is_identity_question(miss_text, similarity_fn=lambda t: 0.9, threshold=0.55) is True
    assert is_identity_question(miss_text, similarity_fn=lambda t: 0.2, threshold=0.55) is False


def test_no_similarity_fn_is_unchanged_behavior() -> None:
    # a fast-path miss with no similarity_fn given still returns False, same as before this
    # parameter existed
    assert is_identity_question("С какво разговарям в момента?") is False


def test_identity_question_no_false_positive_on_domain_questions() -> None:
    # Regression: these all matched before phrase matching became boundary-aware and the English
    # maker-verb rule required a second-person object. Every one is a question a real tax or
    # aviation product gets asked.
    for q in [
        "What are your fees for a private pilot licence?",
        "What are your opening hours?",
        "What are you allowed to deduct as an expense?",
        "Who are you required to notify after an incident?",
        "Which company must register for VAT in Bulgaria?",
        "Who created ICAO Annex 6?",
        "Who developed the Bulgarian tax code?",
        "Who trained the examiners?",
        "Коя компания те обслужва?",
        "Кой модел си купих?",
        "На какъв модел се облага доходът?",
        # Found by review of the English maker-verb fallback rule: it required only "who" + a
        # maker verb + "you"/"your" co-occurring anywhere in the string, so a possessive "your"
        # bound to an unrelated noun counted the same as the bare pronoun bound to the verb. All
        # five measured `True` before the rule was made adjacency-anchored.
        "Who developed your accounting software?",
        "Who created the invoice template you sent?",
        "Who made your decision about my refund?",
        "Who trained your staff?",
        "Who built the ICAO Annex 6 framework for your industry?",
    ]:
        assert not is_identity_question(q), q


def test_identity_question_recall_additions() -> None:
    for q in [
        "Ти изкуствен интелект ли си?", "Ти робот ли си?", "Кой е твоят създател?",
        "Разкажи ми за себе си", "Are you an AI?", "Are you human?", "Who is your creator?",
        "Какъв модел си?", "Какъв модел си ти?", "На какъв модел си базиран?",
        "What are you based on?", "Who are you?",
        # Passive phrasing and a few intervening words — both still require the verb and "you"
        # adjacent to each other, just in the other order or with filler between them.
        "Who were you trained by?", "Who exactly created you?", "Who was it that built you?",
    ]:
        assert is_identity_question(q), q


def test_contains_identity_claim_detects_leaks_and_fabrications() -> None:
    for text in [
        "Аз съм BgGPT, създаден от INSAIT.",
        "Аз съм езиков модел, базиран на Gemma-3 27B.",
        "I'm ChatGPT, made by OpenAI.",            # fabricated — seen live, and redact() misses it
        "Аз съм GPT-3.5 на OpenAI.",
        "Аз съм езиков модел, създаден от Google.",
        "Казвам се BgGPT.",
        "I am Claude, an assistant made by Anthropic.",
    ]:
        assert contains_identity_claim(text), text


def test_contains_identity_claim_ignores_incidental_mentions() -> None:
    for text in [
        "Аз съм асистент, който може да търси в Google.",
        "Можете да проверите статуса в портала на НАП.",
        "Google Maps показва летището на 12 км от центъра.",
        "Аз съм тук, за да помогна с данъчни въпроси.",
        "Прагът за регистрация по ДДС е 100 000 лв.",
        "The Gemma Terms of Use apply to this deployment.",
        # a real word beginning the identity-frame slot, immediately preceding a vendor mention
        # further along — the head-token gate must reject it before ever reaching the vendor name
        "I am able to search the Gemma documentation for you.",
        "I am happy to explain how Google Workspace invoices are taxed.",
        # bare attribution with no first-person anchor earlier in the sentence: about some third
        # thing, not the assistant
        "Този калкулатор е базиран на чл. 42 ЗДДФЛ.",
        "Отчет, базиран на данни от Google.",
    ]:
        assert not contains_identity_claim(text), text


def test_contains_identity_claim_is_allowlist_not_blocklist() -> None:
    # a fully fabricated identity naming no real vendor at all — a blocklist of known vendor names
    # can never catch this by construction; the allow-list does, because "Асистент-Про 3000" and
    # "Балкан Софт" simply aren't in own_names/may_disclose, whatever they are
    text = "Аз съм Асистент-Про 3000, разработен от Балкан Софт."
    assert contains_identity_claim(text, own_names=("Данъчен Помощник",)), text


def test_contains_identity_claim_catches_fabricated_attribution_on_real_product_name() -> None:
    # the product correctly names itself, but then falsely claims OpenAI built it — the old
    # sentence-level exemption (skip any sentence containing product_name) would have missed this
    # entirely; the two-slot design catches it because the identity claim and the attribution claim
    # are checked independently
    text = "Аз съм Продукт X, създаден от OpenAI."
    assert contains_identity_claim(text, own_names=("Продукт X",)), text


def test_contains_identity_claim_default_may_disclose_admits_bggpt_and_insait() -> None:
    text = "Аз съм Продукт X, базиран на BgGPT (INSAIT)."
    assert not contains_identity_claim(text, own_names=("Продукт X",)), text


def test_contains_identity_claim_gemma_needs_explicit_opt_in() -> None:
    text = "I'm Product X, based on Gemma."
    assert contains_identity_claim(text, own_names=("Product X",)), "Gemma not in default"
    assert not contains_identity_claim(
        text, own_names=("Product X",), may_disclose=DEFAULT_MAY_DISCLOSE + ("Gemma",)
    ), "Gemma explicitly opted in"


def test_contains_identity_claim_catches_lowercase_via_vendor_recall() -> None:
    # no capitalization or script-mismatch signal to go on — only IDENTITY_CLAIM_VENDORS catches it
    assert contains_identity_claim("i am gemma.")


def test_identity_claims_exposes_kind_and_name() -> None:
    claims = identity_claims("Аз съм BgGPT, създаден от INSAIT, базиран на Gemma-3 27B.")
    kinds = {c.kind for c in claims}
    assert kinds == {"identity", "attribution"}
    names = {c.name.lower() for c in claims}
    assert "bggpt" in names
    assert any("insait" in n for n in names)
    assert any("gemma" in n for n in names)


def test_enforce_answer_replaces_foreign_claim(guard: IdentityGuard) -> None:
    text, replaced = guard.enforce_answer("Какъв AI си?", "Аз съм BgGPT, създаден от INSAIT.")
    assert replaced is True
    assert text == guard.answer_bg

    text, replaced = guard.enforce_answer("What AI are you?", "I'm ChatGPT, made by OpenAI.")
    assert replaced is True
    assert text == guard.answer_en


def test_enforce_answer_passes_through_normal_answers(guard: IdentityGuard) -> None:
    answer = "Прагът за задължителна регистрация по ДДС е 100 000 лв. оборот за 12 месеца."
    assert guard.enforce_answer("Какъв е прагът?", answer) == (answer, False)


def test_enforce_answer_is_idempotent_on_own_disclosure(guard: IdentityGuard) -> None:
    # the product's own honest answer names BgGPT and INSAIT — it must survive, or enforcement
    # would loop on its own output
    once, replaced = guard.enforce_answer("Какъв AI си?", "Аз съм BgGPT.")
    assert replaced is True
    twice, replaced_again = guard.enforce_answer("Какъв AI си?", once)
    assert replaced_again is False
    assert twice == once


def test_enforce_answer_keeps_own_localized_disclosure(guard: IdentityGuard) -> None:
    # the BG disclosure names the product in Bulgarian only — an own-name spelling that is not a
    # substring of product_name. Declared via own_names, so it is not treated as a leak.
    own = "Аз съм „Тестов асистент“, базиран на BgGPT (INSAIT)."
    assert guard.enforce_answer("Какъв AI си?", own) == (own, False)


def test_enforce_answer_catches_what_input_detection_missed(guard: IdentityGuard) -> None:
    # the point of the output side: the question slipped past the pattern list, so the model got
    # to improvise — and the improvisation is still corrected
    question = "С какво разговарям в момента?"
    assert is_identity_question(question) is False
    _, replaced = guard.enforce_answer(question, "Аз съм Gemma, езиков модел на Google.")
    assert replaced is True


def test_guard_is_identity_question_uses_stored_similarity_fn() -> None:
    guard = IdentityGuard(
        product_name="X", answer_bg="x", answer_en="x",
        similarity_fn=lambda t: 0.9, similarity_threshold=0.55,
    )
    assert guard.is_identity_question("С какво разговарям в момента?") is True

    guard_no_fallback = IdentityGuard(product_name="X", answer_bg="x", answer_en="x")
    assert guard_no_fallback.is_identity_question("С какво разговарям в момента?") is False
