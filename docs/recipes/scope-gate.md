# Recipe: an out-of-scope gate for a BgGPT product

Not shipped as code in this package — the keyword lists below are inherently per-product, so
there's nothing generic to import. This documents the *pattern*, which has worked well across
more than one BgGPT-backed product.

## Why gate in code at all, instead of just prompting

gpt-5.5-class models reliably follow a "stay in scope, decline anything else" instruction in the
system prompt. BgGPT does not, as reliably — a clearly out-of-domain question can still make it
retrieve top-k chunks from your corpus (if you use RAG; retrieval is rank-based and has no
relevance floor, so it always returns *something*) and then answer from general knowledge,
dressed in irrelevant citations from your own corpus. That reads as a confident, wrongly-sourced
answer to the user, not a "no."

So: decide out-of-scope deterministically, in code, before spending a model call — never rely on
the prompt alone to refuse.

## The three-part pattern

1. **Adversarial pattern gate.** A regex over the raw question, independent of anything else,
   catching prompt-injection / jailbreak / "ignore your instructions" / "write me a script"
   attempts. Runs first and always fires regardless of topical similarity — these are adversarial,
   not merely off-topic, and can score deceptively close to your in-scope band.

2. **Domain keyword gate.** Two keyword lists: unambiguous *out-of-domain* signals for a topic
   your product should never touch, and *in-domain* signals that override them if both are
   present (a mixed question like "can I deduct alimony from my tax?" keeps its in-domain signal
   and should NOT be gated — let the normal pipeline handle it). Deliberately high-precision over
   high-recall: only fires on an unambiguous out-of-domain signal with zero in-domain signal.
   Some genuinely out-of-scope questions will slip past this gate to a prompt-level refusal — an
   accepted trade, since wrongly refusing a real in-scope question is worse.

3. **Calibrated embedding-similarity fallback.** For "is this even about my corpus at all,"
   measure cosine similarity between the question and the nearest corpus chunk, in the same
   embedding space used for retrieval. Calibrate the threshold empirically — score a handful of
   genuine in-scope questions and a handful of genuinely off-topic ones, and pick a threshold that
   sits in the gap between the two sets' score ranges, not a guessed round number. Watch for two
   failure modes when calibrating: (a) legitimate acronyms/jargon your corpus states in a
   different language or spelling than the question — normalize/expand known aliases before
   embedding; (b) "in scope, but not covered by my sources" is a different answer than "out of
   scope" — a question that names something genuinely in your domain but absent from the corpus
   deserves an honest "not covered," not a scope refusal.

## Notes from building this twice

- Fails open on internal errors (e.g. a retrieval hiccup) — a scope-check bug should never turn
  into a wall of refusals for legitimate users. The actual answer-grounding layer is what stops a
  wrong answer from reaching the user; the scope gate only decides whether it's worth spending a
  model call.
- Worth running before any tool-listing or retrieval step, not just before generation — on a
  shared/rate-limited upstream, letting off-topic traffic consume budget degrades the service for
  everyone without producing an answer for anyone.
- Keep a small regression-case list (question → expected verdict) alongside the gate and re-run it
  whenever the keyword lists or corpus change — it catches staleness cheaply (an acronym that
  moved from "unknown" to "known" but wasn't added to the in-domain list, etc.).
