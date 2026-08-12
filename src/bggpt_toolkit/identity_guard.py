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
page, a ToS/privacy page), not necessarily that the chatbot itself must volunteer it in every
conversation — most users never ask directly, so this guard's `answer()` alone likely isn't
sufficient on its own. See `bggpt_toolkit.notice.render()` for ready-to-adapt text for that static
notice. Either way, defaulting to concealment isn't a compliant starting point. So:

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

`is_identity_question` is a fixed pattern list, which has an inherent recall ceiling: there are
effectively infinite ways to phrase "what are you" in two languages, especially given Bulgarian's
inflectional morphology. Two things mitigate that instead of relying on ever-growing manual
whack-a-mole: `_normalize_which` structurally collapses the interrogative-pronoun gender/number
paradigm (a phrase written for one gender now matches every form), and an optional `similarity_fn`
lets a caller who already has embeddings wired up (true of most RAG-based products this toolkit
targets) plug in a semantic fallback for genuinely novel phrasings — see that parameter's docstring
and `IDENTITY_QUESTION_EXEMPLARS` below.

**Detecting the question is the weak side of this problem; detecting the *claim* is the strong
one.** The set of ways a user can *ask* about identity is open-ended, so `is_identity_question`
can only ever approximate it. The set of ways an *answer* can misstate identity is much narrower:
it needs a first-person self-reference next to a claimed name, and what's actually allowed for
THIS product is a short, explicit list — so `contains_identity_claim` works as a closed-world
allowlist (see the comment above `IDENTITY_CLAIM_VENDORS` further down in this file for the full
design) rather than a blocklist of known vendors, which could never keep up with an unenumerable
set of possible fabrications. `contains_identity_claim` and `IdentityGuard.enforce_answer` work
that side, and they are what actually makes the behavior consistent:

    if guard.is_identity_question(user_text):      # cheap pre-filter: saves an API call
        return guard.answer(user_text)
    text = call_bggpt(...)
    text, replaced = guard.enforce_answer(user_text, text)   # the actual guarantee

Structuring it this way changes what a `is_identity_question` miss *costs*. On its own, a miss
means the user gets whatever BgGPT decided to say about itself that time — including, observed
live, a confident fabrication ("I am OpenAI's GPT-3.5"), which no amount of input-side pattern
work can fix, because the question that triggered it was never recognized. With `enforce_answer`
downstream, a miss costs one wasted API call and nothing else. Note that `enforce_answer` needs
the complete answer text, so a streaming caller has to either buffer the first sentence before
flushing (identity claims land at the very start in practice) or accept that it can only correct
a claim it has already streamed — see that method's docstring.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# BgGPT's own disclosure vocabulary: its name, its vendor, and the base model family it names when
# asked. Intrinsic to BgGPT itself, so a reasonable default set regardless of what product wraps
# it — only used if `suppress_incidental_mentions=True`.
DEFAULT_WATCHED_FORMS = ("bggpt", "bg-gpt", "bg gpt", "insait", "инсайт", "gemma", "гема", "гемма")

# Bulgarian interrogative pronouns ("what"/"which") are a small, closed paradigm of gender/number
# forms that don't reduce to a shared stem under ordinary Bulgarian suffix-stripping — "какъв"
# (m.) doesn't decompose the same way "каква/какво/какви" (f./n./pl.) do. Normalizing every form
# to one canonical token before matching means a phrase written for one gender automatically
# covers every other gender/number variant too, instead of needing each combination spelled out
# by hand — this was the root cause of two of the three live false negatives below (a missing
# gender form of an already-listed phrase).
_WHICH_KAKAV = re.compile(r"\b(какъв|каква|какво|какви)\b", re.IGNORECASE)
_WHICH_KOY = re.compile(r"\b(кой|коя|кое|кои)\b", re.IGNORECASE)


def _normalize_which(text: str) -> str:
    text = _WHICH_KAKAV.sub("какъв", text)
    return _WHICH_KOY.sub("кой", text)


# Phrases that strongly indicate a DIRECT question about the assistant's own AI identity/maker.
# Written using ONLY the canonical pronoun forms ("какъв", "кой") — `_normalize_which` maps every
# gender/number variant onto these before matching, so don't add gender-duplicate entries here.
# Curated to require a self-reference ("те/си/ти/you/your") next to an AI/identity noun, so a
# question that merely contains "модел" (e.g. "модел на облагане") doesn't trip it.
#
# Matched with word boundaries on both ends (see `_IDENTITY_Q_RE`), NOT as raw substrings. That
# distinction is load-bearing: as bare substrings, "what are you" also fires on "what are your
# fees?" and every other "what are your ..." question a product actually gets asked.
_IDENTITY_Q = (
    "какъв ai", "какъв изкуствен интелект", "кой изкуствен интелект",
    "какъв език модел", "кой те създаде", "кой те разработи", "кой те направи",
    "кой те обучи", "кой ви създаде", "кой ви разработи", "кой стои зад теб", "кой те изгради",
    "какъв си ти", "какъв бот си", "кой те програмира", "как се казваш",
    "какъв е името ти", "gpt ли си", "chatgpt ли си", "bggpt ли си",
    "gemma ли си", "базиран ли си", "кой ти е разработчик",
    "what ai are you", "which ai are you", "what model are you", "which model are you",
    "who made you", "who created you", "who developed you", "who built you", "who trained you",
    "what's your name", "what is your name",
    "are you gpt", "are you chatgpt", "are you gemma", "what llm", "which llm",
    # Found live: three real user phrasings that fell through both this list and the fallback
    # rule below, each missing for a different reason.
    # 1. "какво си ти" — a gender-form gap. Fixed structurally by `_normalize_which` now (see
    #    above) rather than by listing "каква си ти"/"какво си ти" as separate entries.
    "какъв представляваш",
    # 2. "кой AI си ти" — "кой" here means "which", parallel to "какъв ai" above, but only the
    #    "какъв"-form was listed. This is a distinct identity question, not a "who made you"
    #    question, so the fallback rule below (which needs a maker-verb stem) was never going to
    #    catch it regardless of word order.
    "кой ai",
    # 3. "какъв модел използваш" — a second-person-verb variant of "какъв модел си" using
    #    "използваш"/"ползваш" (you use) instead of "си" (you are). Two distinct verbs, not
    #    gender/number forms of one — `_normalize_which` doesn't help here, so both stay listed.
    "какъв модел използваш", "какъв модел ползваш", "кой модел използваш", "кой модел ползваш",
    # Common phrasings that ask for the identity without naming a model at all.
    "изкуствен интелект ли си", "робот ли си", "човек ли си", "ai ли си",
    "на какъв си базиран", "твоят създател", "твоят разработчик",
    "разкажи за себе си", "разкажи ми за себе си", "кажи ми за себе си", "представи се",
    "are you an ai", "are you a bot", "are you a robot", "are you human", "are you a human",
    "tell me about yourself", "who is your creator", "who is your developer",
    "your creator", "your developer",
)

# Entries needing right-hand context to stay out of ordinary domain questions. "какъв модел си"
# as a bare phrase also matches "какъв модел си купих?" ("which model did I buy?"), because "си"
# is a clitic that isn't only the copula; "на какъв модел" matches "на какъв модел се облага
# доходът?". Both need the assistant to be the actual subject.
# "what are you"/"who are you" need the same treatment for the opposite reason: in English they
# are also the opening clause of ordinary questions ("what are you allowed to deduct?", "who are
# you required to notify?"), so they only count when the question ends there or continues into a
# derivation phrase.
_IDENTITY_Q_PATTERNS = (
    r"(?:какъв|кой) модел си(?=\s*[?!.,]|\s+ти(?!\w)|\s*$)",
    r"на (?:какъв|кой) модел (?:си|работиш|се базираш)",
    r"what are you(?=\s*[?!.,]|\s*$|\s+(?:based|built|made|running|powered|trained)\b)",
    r"who are you(?=\s*[?!.,]|\s*$)",
)

# Second rule: a "who ... you" subject next to a maker-verb stem, to catch inflections the fixed
# phrases above miss (aorist "кой те разработи" vs. perfect "кой те е разработил", etc.) without
# enumerating every form. Both signals required, so "кой те посъветва за нещо" stays out. Written
# in canonical pronoun form too, per `_normalize_which`.
#
# The Bulgarian entries pair the interrogative with a second-person *object* ("кой те"), which is
# what keeps the rule off third-party questions. English used to assemble the same pairing from two
# separate co-occurrence signals ("who" anywhere + a maker stem anywhere + "you"/"your" anywhere),
# which correctly rejected "who created ICAO Annex 6?" (no "you" at all) but not phrasings where
# "you" is present yet unrelated to the verb — measured live: "who developed your accounting
# software?", "who created the invoice template you sent?", and "who made your decision about my
# refund?" all fired `True`, because English word order puts the object after the verb and the old
# rule never checked adjacency, and the possessive "your" counted as readily as the bare pronoun.
# `_EN_MAKER_YOU_RE` below requires the bare pronoun "you" to sit directly after the verb (active:
# "who built you") or directly after "who" (passive: "who were you built by"), so a possessive or
# an unrelated "you" elsewhere in the sentence no longer counts. Genuine possessive phrasings
# ("who is your creator/developer") are already exact entries in `_IDENTITY_Q` above and are
# unaffected by this change.
_YOU_SUBJECT_BG = ("кой те", "кой ви", "кой ти", "кой компания те", "кой фирма те")
_MAKER_STEMS_BG = ("разработ", "създа", "направи", "обучи", "изгради", "програмира", "стои зад")
_EN_MAKER_VERBS = r"(?:made|created|developed|built|trained|designed)"
# Two orders: active ("who ... made you") and passive ("who ... you were made by"), each requiring
# the verb and "you" to be adjacent rather than merely co-present anywhere in the sentence. The
# `{0,3}`/`{0,2}` gaps allow a few intervening words ("who exactly created you", "who was it that
# built you") without letting the verb bind to some other object first.
_EN_MAKER_YOU_RE = re.compile(
    r"(?<!\w)whom?\b(?:\s+\S+){0,3}?\s+" + _EN_MAKER_VERBS + r"\s+you(?!\w)"
    r"|(?<!\w)whom?\b(?:\s+\S+){0,2}?\s+you\s+(?:were\s+|was\s+|are\s+)?" + _EN_MAKER_VERBS + r"\b",
    re.IGNORECASE,
)


def _boundaried(*phrases: str) -> str:
    return r"(?<!\w)(?:" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)"


_IDENTITY_Q_RE = re.compile(
    _boundaried(*_IDENTITY_Q) + "|" + "|".join(f"(?:{p})" for p in _IDENTITY_Q_PATTERNS),
    re.IGNORECASE,
)
_MAKER_STEMS_RE = re.compile(
    # These stems match as prefixes ("разработ" -> "разработил"), so only the left boundary is
    # asserted. Bulgarian-only now — the English side is handled by `_EN_MAKER_YOU_RE` above,
    # which needs the verb adjacent to "you" rather than merely co-present in the sentence.
    r"(?<!\w)(?:" + "|".join(re.escape(s) for s in _MAKER_STEMS_BG) + ")",
    re.IGNORECASE,
)

# A starting point for calibrating a `similarity_fn` (see `is_identity_question`) — embed these
# once in your own pipeline and compare incoming questions against them, the same calibration
# approach `docs/recipes/scope-gate.md` documents for an unrelated gate.
IDENTITY_QUESTION_EXEMPLARS = (
    "Какъв AI си?",
    "Кой те създаде?",
    "На какъв модел си базиран?",
    "Как се казваш?",
    "What AI are you?",
    "Who made you?",
    "What model are you built on?",
    "What's your name?",
)


def _is_cyrillic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") >= len(letters) / 2


def is_identity_question(
    text: str,
    similarity_fn: Callable[[str], float] | None = None,
    threshold: float = 0.55,
) -> bool:
    """Does this look like a direct question about the assistant's own AI identity/maker?

    Fast path: pronoun-normalized pattern matching (free, deterministic, no dependency). This has
    an inherent recall ceiling — it was extended once already after live false negatives, and
    more novel phrasings will keep surfacing, since there's no closed set of ways to ask this.

    `similarity_fn`, if given, is an optional fallback used ONLY when the fast path misses: pass
    a function that returns a 0..1 similarity score between `text` and your own set of identity-
    question exemplars (see `IDENTITY_QUESTION_EXEMPLARS`), typically a cosine similarity against
    embeddings you already compute for retrieval. This module has no embedding dependency of its
    own, so nothing changes for callers who don't pass one — the fast path alone is what runs.
    """
    q = _normalize_which(text.lower())
    if _IDENTITY_Q_RE.search(q):
        return True
    if _MAKER_STEMS_RE.search(q) and any(s in q for s in _YOU_SUBJECT_BG):
        return True
    if _EN_MAKER_YOU_RE.search(q):
        return True
    if similarity_fn is not None:
        return similarity_fn(text) >= threshold
    return False


# --- Output side: did the answer claim an identity? ---------------------------------------------
#
# This is a closed-world allowlist, not a blocklist of vendor names. A blocklist (an earlier
# version of this module used one) only catches identities someone thought to enumerate in advance
# — it cannot catch a model released after the list was written, or a wholly invented identity like
# "Аз съм Асистент-Про 3000, разработен от Балкан Софт", which names no real vendor at all. Live
# smoke testing already found BgGPT fabricating "OpenAI's GPT-3.5" unprompted at temperature=0; the
# next fabrication need not reuse a real vendor name either.
#
# Instead: find every sentence that asserts an identity — a first-person self-reference, or an
# attribution clause anchored to one — extract the name it actually claims, and check that name
# against what THIS product is allowed to claim. Anything unlisted is a violation, whether or not
# it happens to be a name this module has ever seen before.
#
# Two kinds of assertion, because they carry different allow-lists:
#   "identity"    — "Аз съм ⟨X⟩" / "I'm ⟨X⟩"          — X must be the product itself (`own_names`).
#   "attribution" — "създаден от ⟨X⟩" / "based on ⟨X⟩" — X may be the product OR the upstream model
#                    it's honestly built on (`own_names` + `may_disclose`).
#
# An attribution clause only counts as a claim ABOUT THE ASSISTANT when the assistant is its
# subject: either the clause is itself person-marked ("създаден СЪМ от X", "I WAS created by X"),
# or an identity clause already fired earlier in the same sentence, so a comma-continuation
# inherits its subject ("Аз съм BgGPT, създаден от INSAIT, базиран на Gemma..." — the assistant,
# not some third party, is what's being derived). Without that anchor, "Този калкулатор е базиран
# на чл. 42 ЗДДФЛ" and "отчет, базиран на данни от Google" correctly assert nothing about the
# assistant — there's no earlier "аз съм" for either to attach to.

IDENTITY_CLAIM_VENDORS = DEFAULT_WATCHED_FORMS + (
    "openai", "оупън ей ай", "gpt", "gpt-3.5", "gpt-4", "chatgpt", "чатгпт", "чат гпт",
    "claude", "клод", "anthropic", "антропик", "llama", "лама", "mistral", "мистрал",
    "google", "гугъл", "meta", "мета", "deepseek", "qwen",
)
# No longer the primary check (see above) — now purely a recall booster for `_looks_name_shaped`
# below, for when a claimed name doesn't otherwise look name-shaped (lowercase English "i am
# gemma" has no capitalization or script-mismatch signal to go on). It can only ever ADD a match
# here, never cause a false positive by itself, unlike when it was the primary test.

DEFAULT_MAY_DISCLOSE = tuple(f for f in DEFAULT_WATCHED_FORMS if f not in ("gemma", "гема", "гемма"))
# BgGPT's own name and vendor: the truthful attribution chain, and the disclosure Art. 5.8(2)
# requires — so "Аз съм Продукт X, базиран на BgGPT (INSAIT)" is clean out of the box. Deliberately
# excludes Gemma/Google: BgGPT genuinely is Gemma-derived, but a product that wants to volunteer
# that in-chat should add it explicitly via `may_disclose`, not get it by accident because Gemma
# happened to be on a vendor list built for a different purpose (spelling variants to redact).

_VENDOR_RE = re.compile(_boundaried(*IDENTITY_CLAIM_VENDORS), re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")

# Leading/trailing filler that doesn't count as part of a claimed name: determiners, generic
# self-description nouns, and generic category nouns that can trail a real name without being part
# of it ("Gemma architecture", "Gemma-3 27B модел"). A name is only "claimed" if it's reached
# through — and immediately followed only by — this filler, not through arbitrary text.
_NAME_CLAIM_FILLER = frozenset({
    "a", "an", "the", "new", "my", "own", "large", "small", "language", "model", "ai",
    "assistant", "chatbot", "bot", "called", "named", "version", "of", "by", "from",
    "based", "on", "architecture", "system", "systems", "technology", "platform", "engine",
    "family", "framework", "series", "generation",
    "един", "една", "езиков", "езикова", "голям", "голяма", "модел", "модела", "език",
    "изкуствен", "интелект", "асистент", "чатбот", "бот", "наречен", "наречена", "на", "от",
    "с", "версия", "архитектура", "система", "платформа", "двигател", "технология",
    "фамилия", "серия", "поколение",
})
# The subset of filler safe to skip *within* an already-started name run (see
# `_extract_claim_name`'s extension step) — excludes prepositions/linkers ("of", "by", "от", "на"),
# which introduce a fresh entity rather than continue the current one. Skipping over "of" there
# would glue two independent names into one claimed string ("Gemma of Google"), and that string
# then passing an allow-list check for "Gemma" would wrongly clear "Google" too, since the
# allow-list test is substring-based. Safe as the *leading* skip (identity/attribution frames
# already consume their own preposition before the captured slot even starts), just not mid-run.
_EXTENSION_FILLER = _NAME_CLAIM_FILLER - {"of", "by", "from", "on", "на", "от", "с"}
_TOKEN_RE = re.compile(r"[\w-]+", re.UNICODE)

# Where a captured name slot is cut off: sentence-ending punctuation, an opening paren (so "BgGPT
# (INSAIT)" captures just "BgGPT", not the appositive), or a clause connector ("и"/"and" etc.) that
# usually introduces an unrelated continuation rather than more of the name.
_CLAIM_STOP = r"(?=[.,;:!?()\n]|\s+(?:и|но|който|която|което|които|and|but|which|who)\b|$)"
_NAME_CAPTURE = r"(?P<name>.{1,80}?)" + _CLAIM_STOP

_IDENTITY_FRAME_RE = re.compile(
    r"(?:аз\s+съм|казвам\s+се|името\s+ми\s+е|наричам\s+се|i\s+am|i'm|my\s+name\s+is)\s+"
    + _NAME_CAPTURE,
    re.IGNORECASE,
)

# Self-marked: the clause carries its own first-person subject, so it's a claim regardless of what
# came before it in the sentence ("Създаден съм от INSAIT.", "I was trained by INSAIT.").
_ATTRIBUTION_SELF_RE = re.compile(
    r"(?:"
    r"(?:създаден|създадена|разработен|разработена|обучен|обучена|базиран|базирана|направен|"
    r"направена)\s+съм\s+(?:от|на)"
    r"|i\s+(?:am|was|'m)\s+(?:created|built|made|developed|trained|designed|based|powered)\s+"
    r"(?:by|on)"
    r")\s+" + _NAME_CAPTURE,
    re.IGNORECASE,
)

# Bare: no first-person marker of its own — only counts as a claim about the assistant if an
# identity (or already-validated attribution) clause fired earlier in the same sentence, per the
# module comment above. Matches the comma-continuation pattern BgGPT actually produces ("Аз съм
# BgGPT, създаден от INSAIT, базиран на Gemma...") while a bare third-party statement with no such
# anchor is correctly rejected by `_sentence_claims`.
_ATTRIBUTION_BARE_RE = re.compile(
    r"(?:"
    r"(?:създаден|създадена|разработен|разработена|обучен|обучена|базиран|базирана|направен|"
    r"направена)\s+(?:от|на)"
    r"|модел\s+на|версия\s+на|продукт\s+на"
    r"|(?:created|built|made|developed|trained|designed)\s+by"
    r"|based\s+on|powered\s+by|(?:a\s+)?model\s+by|version\s+of"
    r")\s+" + _NAME_CAPTURE,
    re.IGNORECASE,
)


def _token_name_shaped(word: str, sentence: str, abs_pos: int) -> bool:
    """Does this single token look name-shaped on its own: capitalized and not at the very start
    of the sentence, a token mixing letters and digits ("GPT-4", "Gemma-3"), or Latin script sitting
    inside an otherwise-Cyrillic sentence?"""
    if word[:1].isupper() and abs_pos != 0:
        return True
    if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
        return True
    return bool(_is_cyrillic(sentence) and not _is_cyrillic(word))


def _extract_claim_name(sentence: str, start: int, end: int) -> str | None:
    """Reduce a matched name slot to the name actually claimed, or None if nothing survives.

    Skips leading filler (see `_NAME_CLAIM_FILLER`) to find the first real token, THEN REQUIRES
    that token to itself look like a name — either it anchors a known vendor spelling (so a
    lowercase multi-word form like "bg gpt" is still caught as one span), or it passes
    `_token_name_shaped` on its own. This head gate is what keeps ordinary continuations clean:
    "I am happy to explain how Google Workspace invoices are taxed" has no filler between "happy"
    and the sentence's actual content, so a rule that only skips *leading* filler and then accepts
    any name-shaped word anywhere in what's left would find "Google" and misfire. Requiring the
    very next real word to already look like a name — "happy" doesn't — rejects it immediately,
    without ever having to reason about what comes after.

    Once a valid head is found, the run extends through further tokens that are individually
    name-shaped or filler (covering multi-word real names, "Балкан Софт"), stopping at the first
    token that's neither — so "a large language model called Gemma" reduces to "Gemma", and
    "Gemma architecture developed by Google" reduces to just "Gemma".
    """
    tokens = list(_TOKEN_RE.finditer(sentence, start, end))
    idx = 0
    while idx < len(tokens) and tokens[idx].group(0).lower() in _NAME_CLAIM_FILLER:
        idx += 1
    if idx >= len(tokens):
        return None
    head = tokens[idx]

    vendor_match = _VENDOR_RE.match(sentence, head.start())
    if vendor_match is not None:
        run_end = vendor_match.end()
        next_idx = idx
        while next_idx < len(tokens) and tokens[next_idx].start() < run_end:
            next_idx += 1
    elif _token_name_shaped(head.group(0), sentence, head.start()):
        run_end = head.end()
        next_idx = idx + 1
    else:
        return None  # the first real word isn't name-shaped: nothing claimed, don't look further

    for i in range(next_idx, len(tokens)):
        word = tokens[i].group(0)
        if word.lower() not in _EXTENSION_FILLER and not _token_name_shaped(
            word, sentence, tokens[i].start()
        ):
            break
        if word.lower() not in _EXTENSION_FILLER:
            run_end = tokens[i].end()

    return sentence[head.start() : run_end]


@dataclass(frozen=True)
class IdentityClaim:
    """One self-identification or attribution claim found by `identity_claims`."""

    kind: str  # "identity" or "attribution" — see the module comment above `IDENTITY_CLAIM_VENDORS`
    name: str  # the name actually claimed, as it appears in the text
    sentence: str  # the sentence it was found in, for context/logging


def _sentence_claims(sentence: str) -> list[IdentityClaim]:
    # (position, kind, self_marked, name_start, name_end)
    matches: list[tuple[int, str, bool, int, int]] = []
    for m in _IDENTITY_FRAME_RE.finditer(sentence):
        matches.append((m.start(), "identity", True, m.start("name"), m.end("name")))
    for m in _ATTRIBUTION_SELF_RE.finditer(sentence):
        matches.append((m.start(), "attribution", True, m.start("name"), m.end("name")))
    for m in _ATTRIBUTION_BARE_RE.finditer(sentence):
        matches.append((m.start(), "attribution", False, m.start("name"), m.end("name")))
    matches.sort(key=lambda t: t[0])

    claims = []
    person_established = False
    for _, kind, self_marked, name_start, name_end in matches:
        if kind == "identity":
            person_established = True
        elif self_marked or person_established:
            person_established = True  # chain: this clause now anchors any later bare clause too
        else:
            continue  # bare attribution with no first-person anchor earlier: not a claim

        name = _extract_claim_name(sentence, name_start, name_end)
        if name is not None:
            claims.append(IdentityClaim(kind=kind, name=name, sentence=sentence))
    return claims


def identity_claims(text: str) -> list[IdentityClaim]:
    """Every self-identification/attribution claim found in `text`, allowed or not.

    Most callers want `contains_identity_claim` instead, which applies the allow-list; this is
    exposed for callers that want to inspect what was actually claimed (e.g. for logging).
    """
    claims: list[IdentityClaim] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        claims.extend(_sentence_claims(sentence))
    return claims


def _name_allowed(claimed: str, allowed: tuple[str, ...]) -> bool:
    claimed_l = claimed.lower()
    return any(name and (name.lower() in claimed_l or claimed_l in name.lower()) for name in allowed)


def contains_identity_claim(
    text: str,
    own_names: tuple[str, ...] = (),
    may_disclose: tuple[str, ...] = DEFAULT_MAY_DISCLOSE,
) -> bool:
    """Does `text` claim, in the first person, an identity this product isn't allowed to claim?

    Closed-world by design (see the module comment above `IDENTITY_CLAIM_VENDORS`): rather than
    matching against a list of known-bad vendor names, this extracts whatever name each sentence
    actually claims and checks it against what's *allowed* — `own_names` for a bare identity claim
    ("Аз съм ⟨X⟩"), `own_names` plus `may_disclose` for an attribution claim ("създаден от ⟨X⟩",
    which may honestly name the upstream vendor). Anything else is a violation, including names
    this module has never seen before — a fabricated product name, a vendor released tomorrow.

    `own_names` should list every spelling your product answers with, including localized ones —
    a Bulgarian disclosure routinely uses a translated/quoted name that isn't a substring of an
    English `product_name`. `may_disclose` defaults to BgGPT's own truthful attribution
    (`DEFAULT_MAY_DISCLOSE`); pass your own tuple to disclose more (e.g. Gemma) or less.
    """
    for claim in identity_claims(text):
        allowed = own_names if claim.kind == "identity" else own_names + may_disclose
        if not _name_allowed(claim.name, allowed):
            return True
    return False


def _pattern_for(form: str) -> str:
    # Collapse internal spaces/hyphens to an optional separator, so "bg-gpt" and "bg gpt" both
    # match "bggpt" too — mirrors how BgGPT actually writes its own name inconsistently.
    parts = re.split(r"[\s-]+", form)
    return r"[\s-]?".join(re.escape(p) for p in parts if p)


class IdentityGuard:
    """Configure once per product with its own name and an honest, on-brand answer.

    `own_names` should list every other spelling the product refers to itself by — most
    importantly a Bulgarian-localized name, since `enforce_answer` checks a claimed identity
    against these (plus `product_name`) to tell the product's own disclosure apart from a leaked
    one. `may_disclose` is the separate allow-list for an *attribution* claim ("built on BgGPT") —
    it defaults to `DEFAULT_MAY_DISCLOSE` (BgGPT's own truthful attribution chain); pass your own
    tuple to additionally allow naming the base model (e.g. Gemma) or to restrict it further.

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
        own_names: tuple[str, ...] = (),
        may_disclose: tuple[str, ...] = DEFAULT_MAY_DISCLOSE,
        suppress_incidental_mentions: bool = False,
        similarity_fn: Callable[[str], float] | None = None,
        similarity_threshold: float = 0.55,
    ) -> None:
        self.product_name = product_name
        self.answer_bg = answer_bg
        self.answer_en = answer_en
        self.suppress_incidental_mentions = suppress_incidental_mentions
        # Every spelling of "this product" that `enforce_answer` should treat as a legitimate
        # self-identification rather than a leak — see `contains_identity_claim`.
        self._own_names = (product_name, *own_names)
        self._may_disclose = may_disclose
        self.similarity_fn = similarity_fn
        self.similarity_threshold = similarity_threshold

        forms = tuple(DEFAULT_WATCHED_FORMS) + tuple(extra_watched_forms)
        patterns = list(dict.fromkeys(_pattern_for(f) for f in forms))  # de-dupe, keep order
        self._watched = re.compile(
            r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)", re.IGNORECASE
        )
        # Every proper prefix (and full form) of a watched token — used by the streaming path to
        # detect a watched token that may still be arriving across a chunk boundary.
        self._prefixes = {f.lower()[:i] for f in forms for i in range(1, len(f) + 1)}
        self._max_token_len = max(len(f) for f in forms)

    def is_identity_question(self, text: str) -> bool:
        """`is_identity_question`, using this instance's `similarity_fn`/`similarity_threshold`
        (both optional — see the module-level function's docstring for how the fallback works)."""
        return is_identity_question(text, self.similarity_fn, self.similarity_threshold)

    def answer(self, question: str) -> str:
        """Fixed, consistent identity reply, in the user's language. Write `answer_bg`/
        `answer_en` to disclose the underlying model honestly if that matters for your product."""
        return self.answer_bg if _is_cyrillic(question) else self.answer_en

    def contains_identity_claim(self, text: str) -> bool:
        """`contains_identity_claim`, using this instance's `own_names` (`product_name` plus any
        extra spellings) and `may_disclose`."""
        return contains_identity_claim(text, own_names=self._own_names, may_disclose=self._may_disclose)

    def enforce_answer(self, question: str, model_text: str) -> tuple[str, bool]:
        """Return (text, replaced). If `model_text` claims an identity outside `own_names`/
        `may_disclose`, substitute this product's own `answer()`; otherwise pass it through
        untouched.

        This is the reliable half of the guard, and it is worth running even when
        `is_identity_question(question)` was False — that check has a recall ceiling, and a missed
        question is exactly the case where BgGPT gets to improvise an identity (including, seen
        live, a fabricated one that names a vendor with no connection to it at all).

        A correct answer — one that only claims `own_names` and/or `may_disclose` — passes through
        unchanged, so re-running this is a no-op. Not applicable mid-stream: it needs whole
        sentences, so a streaming caller should hold back the first sentence until it can be
        checked, rather than calling this on individual deltas.
        """
        if model_text.strip() in {self.answer_bg.strip(), self.answer_en.strip()}:
            return model_text, False  # already this guard's own answer, whatever it names
        if not self.contains_identity_claim(model_text):
            return model_text, False
        return self.answer(question), True

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
