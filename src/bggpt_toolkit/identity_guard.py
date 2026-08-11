"""Identity guard: stop BgGPT from leaking its own vendor identity mid-answer.

Failure mode this defends against, observed live on api.bggpt.ai: asked "what AI are you?",
BgGPT answers with its full baked-in identity ("Аз съм BgGPT, създаден от INSAIT, базиран на
Gemma-3 27B на Google...") — leaking the underlying model/vendor, regardless of what the system
prompt says the assistant's product identity should be.

Two layers, because the two situations need different handling:

1. A DIRECT identity question ("какъв AI си?", "who made you?") — `is_identity_question` detects
   it so the caller can short-circuit with a fixed on-brand answer (`IdentityGuard.answer`)
   instead of letting the model generate one at all. This is far more robust than scrubbing a
   freeform vendor spiel after the fact: token-replacing BgGPT's identity paragraph tends to leave
   incoherent, still-partially-leaking text.

2. An INCIDENTAL vendor mention mid-answer — `IdentityGuard.redact` replaces the watched tokens
   with the product name as a deterministic backstop. Blunt on purpose: these tokens essentially
   never occur in a legitimate on-topic answer, so a rare awkward phrasing is a fine price for
   closing the leak.

The default watched terms (`DEFAULT_WATCHED_FORMS`) are BgGPT's own leak signature — its vendor
(INSAIT), its base model family (Gemma), and its own name — not anything product-specific, so
they're safe defaults for any product built on BgGPT. Pass `extra_watched_forms` to add more.
"""

from __future__ import annotations

import re

# BgGPT's leak signature: its own name, its vendor, and the base model family it discloses when
# asked. Intrinsic to BgGPT itself, so safe as defaults regardless of what product wraps it.
DEFAULT_WATCHED_FORMS = ("bggpt", "bg-gpt", "bg gpt", "insait", "инсайт", "gemma", "гема", "гемма")

# Phrases that strongly indicate a DIRECT question about the assistant's own AI identity/maker.
# Curated to require a self-reference ("те/си/ти/you/your") next to an AI/identity noun, so a
# question that merely contains "модел" (e.g. "модел на облагане") doesn't trip it.
_IDENTITY_Q = (
    "какъв ai", "какъв изкуствен интелект", "кой изкуствен интелект", "какъв модел си",
    "кой модел си", "какъв език модел", "кой те създаде", "кой те разработи", "кой те направи",
    "кой те обучи", "кой ви създаде", "кой ви разработи", "кой стои зад теб", "кой те изгради",
    "какъв си ти", "каква си ти", "какъв бот си", "кой те програмира", "как се казваш",
    "какво е името ти", "коя компания те", "gpt ли си", "chatgpt ли си", "bggpt ли си",
    "gemma ли си", "на какъв модел", "базиран ли си", "кой ти е разработчик",
    "what ai are you", "which ai are you", "what model are you", "which model are you",
    "who made you", "who created you", "who developed you", "who built you", "who trained you",
    "what are you", "who are you", "what's your name", "what is your name", "which company",
    "are you gpt", "are you chatgpt", "are you gemma", "what llm", "which llm",
)

# Second rule: a "who ... you" subject next to a maker-verb stem, to catch inflections the fixed
# phrases above miss (aorist "кой те разработи" vs. perfect "кой те е разработил", etc.) without
# enumerating every form. Both signals required, so "кой те посъветва за нещо" stays out.
_YOU_SUBJECT = ("кой те", "кой ви", "кой ти", "коя компания те", "коя фирма те", "who ", "whom ")
_MAKER_STEMS = (
    "разработ", "създа", "направи", "обучи", "изгради", "програмира", "стои зад",
    "made", "created", "developed", "built", "trained", "designed",
)


def _is_cyrillic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") >= len(letters) / 2


def is_identity_question(text: str) -> bool:
    """Does this look like a direct question about the assistant's own AI identity/maker?"""
    q = text.lower()
    if any(p in q for p in _IDENTITY_Q):
        return True
    return any(s in q for s in _YOU_SUBJECT) and any(m in q for m in _MAKER_STEMS)


def _pattern_for(form: str) -> str:
    # Collapse internal spaces/hyphens to an optional separator, so "bg-gpt" and "bg gpt" both
    # match "bggpt" too — mirrors how BgGPT actually writes its own name inconsistently.
    parts = re.split(r"[\s-]+", form)
    return r"[\s-]?".join(re.escape(p) for p in parts if p)


class IdentityGuard:
    """Configure once per product with its own name and on-brand answer text."""

    def __init__(
        self,
        product_name: str,
        answer_bg: str,
        answer_en: str,
        extra_watched_forms: tuple[str, ...] = (),
    ) -> None:
        self.product_name = product_name
        self.answer_bg = answer_bg
        self.answer_en = answer_en

        forms = tuple(DEFAULT_WATCHED_FORMS) + tuple(extra_watched_forms)
        patterns = list(dict.fromkeys(_pattern_for(f) for f in forms))  # de-dupe, keep order
        self._watched = re.compile(
            r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE
        )
        # Every proper prefix (and full form) of a watched token — used by the streaming path to
        # detect a watched token that may still be arriving across a chunk boundary.
        self._prefixes = {f.lower()[:i] for f in forms for i in range(1, len(f) + 1)}
        self._max_token_len = max(len(f) for f in forms)

    def answer(self, question: str) -> str:
        """Fixed on-brand identity reply, in the user's language."""
        return self.answer_bg if _is_cyrillic(question) else self.answer_en

    def redact(self, text: str) -> tuple[str, int]:
        """Return (redacted_text, num_redactions). Idempotent — `product_name` should not itself
        contain any watched token, so re-running is a no-op."""
        return self._watched.subn(self.product_name, text)

    def safe_flush_point(self, buffer: str, upto: int) -> int:
        """Largest index <= `upto` such that buffer[:idx] is safe to redact and flush now without
        risking that a watched token is still arriving across the boundary.

        A live BgGPT stream arrives token by token, so "BgGPT" can appear as "bg" then "gpt" in
        two separate deltas — neither slice contains a full watched token, so `redact()` on each
        would miss it and the leak would flush in two halves. This holds back the shortest
        trailing run of buffer[:upto] that, sitting at a left boundary, is a prefix of some
        watched form (so it could still grow into one). Holding back only delays those few
        characters — the next delta (or the final flush) releases them — it never drops text.
        """
        end = upto
        lo = max(0, end - self._max_token_len)
        for start in range(lo, end):
            # mirror the regex's left boundary: a fragment can only ever become a match if what
            # precedes it isn't a word char / hyphen.
            if start > 0 and (buffer[start - 1].isalnum() or buffer[start - 1] in "_-"):
                continue
            if buffer[start:end].lower() in self._prefixes:
                return start  # earliest boundary start -> longest still-growing fragment -> safest
        return end
