# Plan: Token-Efficient Word Encoding for UUIDv4

## Problem

A canonical UUIDv4 string (`550e8400-e29b-41d4-a716-446655440000`) is 36
characters of hex and hyphens. OpenAI's BPE tokenizers don't have merge
rules tuned for arbitrary hex runs, so a UUID typically costs **18-23
tokens** depending on encoding — expensive when UUIDs appear repeatedly in
prompts, tool outputs, or long agent transcripts.

Goal: a deterministic, reversible function that maps a UUIDv4 to a short
sequence of real English words, where each word is guaranteed to be
exactly one token under the target tokenizer, so the encoded form costs
fewer tokens than the raw UUID while staying human-readable.

## Prior art (investigated before writing this plan)

1. **[`id-token-nicer`](https://github.com/thejens/id-token-nicer)**
   (Rust + PyO3 bindings, ~7 commits, 1 star) already does this and more:
   bit-to-word mapping via a 4-round Feistel network + xxHash64 (so the
   wordlist never has to be shipped — decoding is pure arithmetic),
   checksums, bloom filters for typo detection, and wordlists tuned for
   single-token status across cl100k_base, o200k_base, Gemma, and Llama 3
   simultaneously. Their reported numbers: raw hex UUID ≈18-23 tokens;
   their token-optimized word mode ≈17-18 tokens; a separate non-word
   "numeric mode" (raw 128-bit integer as a decimal string) gets to ~13
   tokens (~40% savings) but isn't word-based.
2. **[BAML's "Using UUIDs in prompts is bad"](https://boundaryml.com/blog/uuid-swap)**
   makes an orthogonal point: token count isn't the only problem — models
   make outright transcription errors reading/reproducing raw UUIDs. Their
   fix is non-bijective: swap UUIDs for small session-local integer
   aliases before the prompt and swap back after, rather than a reversible
   global encoding.

**Conclusion:** the technique is not novel, but it has no mainstream,
widely-adopted, pure-Python implementation. `id-token-nicer` is a good
reference for the algorithmic idea (word-per-bit-chunk, single-token
filtering) but is Rust-based, unproven, and more complex than we need
(Feistel mixing, checksums, bloom filters, multi-tokenizer intersection).

**Decisions made with the user** (see conversation): build a simpler,
from-scratch, pure-Python implementation; target UUIDv4 only (strip the 6
fixed version/variant bits, encode the 122 random bits); optimize the
wordlist for `o200k_base` only (current GPT-4o/GPT-5-family encoding);
this document is the deliverable — implementation is a follow-up.

## Algorithm

### 1. Bit extraction (UUIDv4-specific)

A UUIDv4's 128 bits contain 6 fixed bits: the 4-bit version nibble
(`0100`, bits 48-51) and the 2-bit variant (`10`, top 2 bits of byte 8).
Stripping these leaves **122 bits of payload**, call it `P`
(`0 <= P < 2**122`).

- `encode`: parse the UUID, assert version == 4 and variant is RFC 4122,
  remove the 6 fixed bits, pack the remaining bits into an integer `P`.
- `decode`: reverse — reinsert the fixed version/variant bits at their
  canonical positions and format as a standard UUID string.

Non-v4 input to `encode` raises `ValueError` (explicitly out of scope per
the "UUIDv4-only" decision).

### 2. Integer → words

Pick a wordlist size `V` (a power of two, so each word carries a fixed
number of bits `b = log2(V)`). Word count `W = ceil(122 / b)`:

| V (words) | bits/word | words needed (W) |
|-----------|-----------|-------------------|
| 2048      | 11        | 12                |
| 4096      | 12        | 11                |
| 8192      | 13        | 10                |
| 16384     | 14        | 9                 |

`V` is **not fixed by this plan** — it's determined empirically in
Implementation Step 1 by how many genuinely single-token, unambiguous,
spellable English words `o200k_base` actually offers (see below). Bigger
`V` means fewer words but a thinner, more obscure candidate pool to draw
from; the implementation should pick the largest `V` that still leaves a
comfortable margin of "boring, easy to read/type/say" words.

Encoding is then a straightforward base-`V` positional representation of
`P` across `W` digits (via repeated `divmod`), each digit mapped to
`WORDLIST[digit]`. Decoding reverse-maps each word to its index via a
`dict[str, int]`, reconstructs `P`, and validates `0 <= P < 2**122`
(out-of-range values — e.g. from a mistyped word combination — are
rejected, which incidentally catches some but not all typos; a real
checksum is explicitly a stretch goal, not MVP, per the "simpler" choice).

### 3. Separator / output format

This needs empirical verification, not assumption: joining words with a
space is the safe default (BPE vocabularies commonly include a
leading-space variant of common words as its own single token, so `"orbit
dodge cactus"` should tokenize as one token per word). A hyphen separator
(`"orbit-dodge-cactus"`) is more convenient as an identifier/slug but risks
the tokenizer merging the hyphen into a neighboring word as a compound
token, silently breaking the single-token-per-word guarantee — Step 1 below
must measure this before picking a default. Plan: implement both
`to_phrase()` (space-joined, token-optimal, primary form) and `to_slug()`
(hyphen-joined, identifier-safe) and let the benchmark decide which one
ships as the default `encode_uuid4()` output.

## Wordlist construction

1. Source a candidate corpus of common, unambiguous English words (e.g. a
   frequency-ranked wordlist rather than a raw system dictionary, to avoid
   obscure/archaic entries).
2. Filter to lowercase, ASCII-only, alphabetic tokens within a sane length
   range (long/short outliers tend to be either abbreviation-prone or
   token-inefficient).
3. Run each candidate through `tiktoken.get_encoding("o200k_base")` and
   keep only words that encode to **exactly one token** as a standalone,
   space-prefixed string (matching how it will actually appear in the
   joined phrase).
4. Apply a profanity/offensive-word blocklist (reuse an existing list,
   e.g. LDNOOBW, rather than hand-curating).
5. Deduplicate, sort deterministically (for reproducible indices), and
   truncate/select to the chosen power-of-two size `V`.
6. Write the result as a static, checked-in Python data file — a plain
   `tuple[str, ...]` — so the runtime codec has **no tiktoken dependency**
   and no network dependency. Only the *generator script* and the
   *wordlist-integrity test* need `tiktoken`.

## Module layout

**Superseded — see "PyPI extraction and OpenAI SDK integration" below.**
This section documents the layout as of the intermediate step, when the
implementation was a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
member of this repo, published standalone to PyPI as `uuid4-words`, with a
thin re-export inside this repo for existing in-repo imports. It has since
moved out of this repo entirely; kept here for the historical record.

```
packages/uuid4-words/            # standalone package -- source of truth, PyPI: uuid4-words
    pyproject.toml
    LICENSE                       # MIT
    README.md                     # PyPI long description; OpenAI SDK integration docs
    src/uuid_words/
        __init__.py                # public API: encode_uuid4, decode_uuid4
        codec.py                   # bit extraction + base-V <-> words, no tiktoken import
        wordlist.py                # generated: WORDS: tuple[str, ...], committed to git
        integrations/openai.py     # WordsUUID (pydantic type) + with_uuid_words decorator
    tests/
        test_codec.py               # round-trip, token-count benchmark, edge cases,
        test_integrations_openai.py # wordlist integrity (tiktoken-gated, see risk below)

src/embeddings_space/uuid_words/  # thin re-export of the above, for this repo's own code
    __init__.py
    codec.py
    wordlist.py

scripts/
    build_uuid_wordlist.py   # offline generator (corpus -> filtered -> wordlist.py)
                              # writes to packages/uuid4-words/src/uuid_words/wordlist.py
                              # dev-only; requires tiktoken + network access
                              # to fetch the o200k_base merge file once

tests/
    test_uuid_words.py   # in-repo integration test against the embeddings_space shim
```

Public API sketch:

```python
def encode_uuid4(value: uuid.UUID | str) -> str: ...
def decode_uuid4(words: str) -> uuid.UUID: ...
```

Raises `ValueError` for: non-v4 input to `encode_uuid4`, wrong word count,
an unrecognized word, or a decoded payload out of the valid `2**122` range.

`tiktoken` is added as a **dev/test dependency only** (`pyproject.toml`
`[dependency-groups] dev` or `[project.optional-dependencies]`), not a
runtime dependency of the package.

## Testing plan

- **Round-trip**: `encode_uuid4(decode_uuid4(x)) == x` and vice versa over
  thousands of randomly generated `uuid4()` values.
- **Token-count benchmark**: for a sample of UUIDs, compare
  `len(enc.encode(str(u)))` (raw) against `len(enc.encode(encoded_form))`
  using `o200k_base`; assert a measurable reduction and report the average
  % savings. This is the test that validates the whole premise of the
  project — it should also run for both `to_phrase()` and `to_slug()` to
  settle the separator question.
- **Wordlist integrity**: every entry in `WORDS` is a single `o200k_base`
  token, `len(WORDS)` is the expected power of two, and there are no
  duplicates. Gate this test behind tiktoken/network availability (see
  risk below) so it's meaningful in CI but doesn't block environments
  without that access.
- **Edge cases**: version/variant bit round-trip on known UUIDs, malformed
  input (wrong word count, unknown word, non-v4 UUID) all raise clear
  errors.

## Known risk / blocker for implementation

`tiktoken` downloads its BPE merge tables from
`openaipublic.blob.core.windows.net` on first use — this session's sandbox
egress proxy blocks that host, so wordlist generation and the tiktoken-gated
tests could not be run or verified numerically while writing this plan.
Options for the implementation step: (a) generate the wordlist once in an
environment with broader network access and commit the static result (the
runtime package never needs network access again), or (b) vendor a copy of
the `o200k_base.tiktoken` merge file into the repo/CI cache. Either way,
this needs to happen before Implementation Step 1 can produce real numbers.

## Out of scope (per decisions made)

- UUID versions other than v4 (v1, v5, v7, nil, etc.)
- `cl100k_base` or other non-OpenAI tokenizers
- Checksums / bloom-filter typo detection (BAML/`id-token-nicer`-style
  robustness) — flagged as a possible Phase 2, not MVP
- Depending on or wrapping `id-token-nicer`

## Implementation steps (follow-up work, not part of this plan's deliverable)

1. Get `tiktoken` network access sorted (see risk above); build the
   candidate word corpus and run the single-token filter against
   `o200k_base`; empirically decide `V` (wordlist size) from how many
   usable words survive filtering.
2. Implement `codec.py` (pure Python, no tiktoken import) with the bit
   extraction and base-`V` word mapping.
3. Generate and commit `wordlist.py` via `scripts/build_uuid_wordlist.py`.
4. Write `tests/test_uuid_words.py` covering round-trip, benchmark, and
   integrity cases above.
5. Run the token-count benchmark for real, decide the default separator,
   and record actual savings numbers (replacing the estimates in this doc).

## Implementation status

All 5 implementation steps are done. `src/embeddings_space/uuid_words/`
(`codec.py`, `wordlist.py`, `__init__.py`), `scripts/build_uuid_wordlist.py`,
and `tests/test_uuid_words.py` are all in the repo, and `uv run pytest
tests/` passes (15 passed, 0 skipped).

**Step 1 and step 5 are now complete.** A later session had network access
to `openaipublic.blob.core.windows.net`, so `scripts/build_uuid_wordlist.py`
was re-run with real `tiktoken`:

- The committed `wordlist.py` is now **VERIFIED**: all 2048 words are
  confirmed single-token under `o200k_base` (as `" " + word`), not just
  frequency-filtered candidates.
- `tests/test_uuid_words.py::TestTokenSavings` runs for real (no longer
  skipped) and passes: `test_wordlist_is_single_token` and
  `test_phrase_uses_fewer_tokens_than_raw_uuid` both confirmed.
- **Real token-count benchmark** (200 random UUIDv4s, `o200k_base`):
  - raw UUID string: avg 22.8 tokens
  - `encode_uuid4` (space-joined phrase): avg 12.0 tokens — **~47% savings**
  - `encode_uuid4_slug` (hyphen-joined): avg 20.1 tokens — **~12% savings**

  The hyphen separator loses most of the benefit because o200k_base's BPE
  merges hyphens into neighboring words (e.g. `raise-review` doesn't split
  the way `raise review` does), confirming the risk flagged in the
  "Separator / output format" section above.
- **Separator decision**: `encode_uuid4` (space-joined) is the default/
  primary form, per the original plan; `encode_uuid4_slug` remains
  available for identifier/URL/filename contexts where hyphens are
  required, with its savings clearly documented as smaller.
- Notably, the ~47% savings for the space-joined form beats the ~17-18
  token estimate reported for `id-token-nicer`'s word mode (see "Prior
  art" above), and even beats their non-word numeric-mode estimate
  (~13 tokens, ~40% savings) — likely because this wordlist was filtered
  for single-token status specifically against `o200k_base`, whereas
  `id-token-nicer` optimizes across multiple tokenizers simultaneously.

## PyPI extraction and OpenAI SDK integration

The implementation was first extracted into `packages/uuid4-words/` as a
standalone uv workspace member of this repo (zero runtime dependencies,
MIT-licensed), then fully moved out into its own repo:
[`EdOlson-Morgan/uuid4-words`](https://github.com/EdOlson-Morgan/uuid4-words),
intended for publication to PyPI as
[`uuid4-words`](https://pypi.org/project/uuid4-words/). This repo
(`shared-embeddings-space-experiments`) no longer contains the
implementation or depends on it — `packages/uuid4-words/` and
`src/embeddings_space/uuid_words/` were both removed rather than re-linked,
per the decision made when the standalone repo was created. This doc
remains here as the historical design record; the canonical code, tests,
README, and release tooling now live in the `uuid4-words` repo.

Two OpenAI Python SDK integration recipes are documented and implemented in
that repo's `uuid_words.integrations.openai` (see its README for the full
write-up): a `WordsUUID` Pydantic field type for UUIDs that flow through
structured outputs / tool calling, and a `with_uuid_words` decorator +
`extract_uuid_words` for UUIDs embedded in freeform prompt/transcript text.
The SDK itself has no middleware/plugin hook for this, so both are
explicit, opt-in patterns rather than a global request/response rewriter —
see that README's "Why not a global request/response rewriter?" section for
why blind text-scanning was rejected.

Release process (tagging, PyPI Trusted Publisher setup, the tag-triggered
`publish.yml` GitHub Actions workflow) is documented in that repo's
`PUBLISHING.md`. As of this writing the package has not yet actually been
published — that requires a human to complete the one-time PyPI
account/Trusted Publisher setup described there.
