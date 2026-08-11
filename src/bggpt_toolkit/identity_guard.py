"""Identity guard: help a product handle BgGPT's inconsistent persona-override behavior.

Observed live on api.bggpt.ai: even when a system prompt sets a custom product persona, asked
"what AI are you?" BgGPT sometimes answers with its own baked-in identity instead ("Аз съм BgGPT,
създаден от INSAIT, базиран на Gemma-3 27B..."). This is not a security flaw or a secret being
exposed — BgGPT is INSAIT's own open, publicly attributed model, and INSAIT plainly wants it known
as BgGPT. It's a persona-override reliability gap: the model doesn't consistently keep to a custom
identity you've asked it to adopt, one way or the other, depending on phrasing and temperature.

**Default behavior here is to disclose, not conceal.** INSAIT's own Terms of Service for the API
(https://bggpt.ai/terms, Art. 5.8(2)) require that you "explicitly notify End Users that the
applications/services/products they access are based on the BgGPT Model" — this is a stated term
of using the API, not an inference. Accepting those terms also binds you to Google's Gemma Terms
of Use (Art. 1.7), whose Prohibited Use Policy separately restricts misleading claims of expertise
or capability in sensitive areas — health, finance, government services, legal.

Note that Art. 5.8(2) is about the *product* giving End Users notice somewhere (a footer, an about
page, a ToS/privacy page — see e.g. how a project might document this in its own "rights and model
used" section), not necessarily that the chatbot itself must volunteer it in every conversation.
This guard's `answer()` is one legitimate way to satisfy it conversationally; a static notice
elsewhere works too. Either way, defaulting to concealment isn't a compliant starting point. So:

1. A DIRECT identity question ("какъв AI си?", "who made you?") — `is_identity_question` detects
   it so the caller can short-circuit with a fixed, honest answer (`IdentityGuard.answer`) instead
   of leaving it to chance what the model says. Write `answer_bg`/`answer_en` to disclose plainly,
   e.g. "I'm Продукт X, built on BgGPT (INSAIT)." — this guard only makes the answer consistent,
   it doesn't tell you to hide anything.

2. An INCIDENTAL vendor mention mid-answer is left alone by default. `IdentityGuard.redact` is an
   *opt-in* backstop (`suppress_incidental_mentions=True`) for products that have made their own
   informed call to suppress incidental mentions — it's blunt on purpose when enabled, since these
   tokens essentially never occur in a legitimate on-topic answer.

`DEFAULT_WATCHED_FORMS` names BgGPT's own vendor (INSAIT), its base model family (Gemma), and its
own name — intrinsic to BgGPT itself, not to whatever product wraps it. Pass `extra_watched_forms`
to add more, only relevant if you enable `suppress_incidental_mentions`.
"""

from __future__ import annotations

import re

# BgGPT's own disclosure vocabulary: its name, its vendor, and the base model family it names when
# asked. Intrinsic to BgGPT itself, so a reasonable default set regardless of what product wraps
# it — only used if `suppress_incidental_mentions=True`.
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
    """Configure once per product with its own name and an honest, on-brand answer.

    `suppress_incidental_mentions` defaults to False (disclose): `redact()` and
    `safe_flush_point()` are no-ops unless you explicitly opt in. See the module docstring for
    why disclosure is the default rather than concealment.
    """

    def __init__(
        self,
        product_name: str,
        answer_bg: str,
        answer_en: str,
        extra_watched_forms: tuple[str, ...] = (),
        suppress_incidental_mentions: bool = False,
    ) -> None:
        self.product_name = product_name
        self.answer_bg = answer_bg
        self.answer_en = answer_en
        self.suppress_incidental_mentions = suppress_incidental_mentions

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
        """Fixed, consistent identity reply, in the user's language. Write `answer_bg`/
        `answer_en` to disclose the underlying model honestly if that matters for your product."""
        return self.answer_bg if _is_cyrillic(question) else self.answer_en

    def redact(self, text: str) -> tuple[str, int]:
        """Return (text, num_redactions). A no-op unless `suppress_incidental_mentions=True` —
        see the class docstring. When enabled, idempotent: `product_name` should not itself
        contain a watched token, so re-running is a no-op."""
        if not self.suppress_incidental_mentions:
            return text, 0
        return self._watched.subn(self.product_name, text)

    def safe_flush_point(self, buffer: str, upto: int) -> int:
        """Largest index <= `upto` such that buffer[:idx] is safe to flush now without risking
        that a watched token, still arriving across a delta boundary, gets redacted in two halves.

        A no-op (returns `upto` unchanged) unless `suppress_incidental_mentions=True` — with
        nothing being redacted, there is nothing to protect a boundary for.

        When enabled: a live BgGPT stream arrives token by token, so "BgGPT" can appear as "bg"
        then "gpt" in two separate deltas — neither slice contains a full watched token, so
        `redact()` on each would miss it. This holds back the shortest trailing run of
        buffer[:upto] that, sitting at a left boundary, is a prefix of some watched form (so it
        could still grow into one). Holding back only delays those few characters — the next
        delta (or the final flush) releases them — it never drops text.
        """
        if not self.suppress_incidental_mentions:
            return upto
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
