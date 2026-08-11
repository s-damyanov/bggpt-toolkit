# Recipe: better input-side identity pre-filters than the built-in phrase list

`is_identity_question` is a fixed pattern list, kept in the package as a zero-dependency
fallback — it needs nothing else wired up, so it always works. But it is not the reliable part of
identity handling and shouldn't be treated as one; `IdentityGuard.enforce_answer()` is (see the
README's "Identity guard" section and `identity_guard.py`'s module docstring for why the two
sides of this problem aren't symmetric). Once `enforce_answer` is wired in, a missed input-side
question only costs one wasted API call, not a wrong answer — so it's worth spending a little
effort improving recall here, but not worth chasing every phrasing by hand.

If your product already has one of the two things below, prefer it over growing the phrase list.

## 1. Reuse your existing scope gate

In a RAG product, an identity question ("what are you?", "who made you?") is just a special case
of *out-of-corpus*: it retrieves nothing relevant, and a retrieval-similarity signal you likely
already compute (see [`scope-gate.md`](scope-gate.md)) tells you that for free. Wiring the same
signal in as `is_identity_question`'s `similarity_fn` means zero identity-specific keyword
maintenance — the scope gate's own calibration does the work.

```python
def scope_similarity(text: str) -> float:
    # whatever your scope gate already computes: 1 - (distance to nearest corpus chunk), etc.
    return 1.0 - nearest_corpus_distance(text)

guard = IdentityGuard(
    product_name="My Assistant",
    answer_bg="...", answer_en="...",
    similarity_fn=scope_similarity,
    similarity_threshold=0.6,  # calibrate against your own in-scope/out-of-scope examples
)
```

This works well when identity questions and genuinely off-topic questions land in a similar
region of your embedding space relative to your corpus — usually true, since neither is *about*
your domain.

## 2. Exemplar-embedding similarity

If there's no existing scope gate, but the product already computes embeddings for retrieval,
embed `identity_guard.IDENTITY_QUESTION_EXEMPLARS` once and compare incoming questions against
them directly:

```python
from bggpt_toolkit.identity_guard import IDENTITY_QUESTION_EXEMPLARS

exemplar_vectors = [embed(q) for q in IDENTITY_QUESTION_EXEMPLARS]

def identity_similarity(text: str) -> float:
    v = embed(text)
    return max(cosine_similarity(v, ev) for ev in exemplar_vectors)

guard = IdentityGuard(..., similarity_fn=identity_similarity, similarity_threshold=0.55)
```

Calibrate the threshold the same way `scope-gate.md` recommends: score a handful of real identity
questions and a handful of genuinely unrelated ones, and pick a threshold in the gap between the
two sets' score ranges.

## When to fall back to the built-in phrase list

Neither of the above, and no embeddings pipeline to hang a `similarity_fn` off of at all — that's
the only case where growing `_IDENTITY_Q` by hand is the remaining option, and even then: don't.
The phrase list has already been extended twice after live false negatives and will keep missing,
by construction (see the module docstring — there's no closed set of ways to ask "what are you"
in two languages). Spend the effort on `enforce_answer`'s allow-list instead
(`own_names`/`may_disclose`) — that's the side of this problem that's actually closed, and where
effort compounds instead of leaking away into an unenumerable phrase list.
