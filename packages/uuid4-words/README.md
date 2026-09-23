# uuid4-words

Deterministic, reversible encoding of a UUIDv4 as a short phrase of plain
English words instead of raw hex — so it costs fewer LLM tokens in prompts
and transcripts, while staying human-readable.

```python
from uuid_words import encode_uuid4, decode_uuid4

encode_uuid4("173c6f57-2c5f-41a3-b898-f49eae04ffcc")
# "ability content shared prime queen think dude created fact frank bay wine"

decode_uuid4("ability content shared prime queen think dude created fact frank bay wine")
# UUID('173c6f57-2c5f-41a3-b898-f49eae04ffcc')
```

Measured against OpenAI's `o200k_base` tokenizer (200 random UUIDv4s): a raw
UUID string costs an average of **22.8 tokens**; the word-phrase form above
costs **12.0** — a **~47% reduction**. Every word in the shipped wordlist is
independently verified to be exactly one `o200k_base` token, so that ratio
holds regardless of which UUID you encode.

A hyphenated `encode_uuid4_slug()` form is also provided for
filenames/URLs/identifiers, but note it only saves ~12% — BPE merges the
hyphens into neighboring words, so most of the single-token guarantee is
lost. Prefer `encode_uuid4` (space-joined) whenever token count matters.

## Install

```bash
pip install uuid4-words
```

No runtime dependencies. Requires Python >= 3.9.

## How it works

A UUIDv4 has 122 bits of actual randomness (the other 6 are the fixed
version/variant bits). Those 122 bits are packed into a base-2048 integer —
2048 being the size of the shipped wordlist, chosen so each digit is exactly
11 bits — and each digit maps to one word. Decoding reverses the process.
The wordlist itself is a static, checked-in Python tuple: no network access
or tokenizer dependency at runtime, only when regenerating it.

Full design writeup, prior-art comparison, and the benchmark methodology:
[`docs/uuid-token-optimization-plan.md`](https://github.com/EdOlson-Morgan/shared-embeddings-space-experiments/blob/main/docs/uuid-token-optimization-plan.md)
in the source repo.

## Using this with the OpenAI Python SDK

The `openai` SDK has no middleware/plugin hook for rewriting message or
tool-call content — it's a generated REST client, not a framework with
callback hooks. So there's no single "decorator that plugs it in"; instead,
pick one of the two patterns below depending on **where your UUIDs actually
appear**. Both are opt-in (nothing here monkeypatches the SDK), and both
avoid scanning arbitrary text for "11 words in a row that happen to be in
the wordlist" — that's not a safe signal, since these are common English
words.

### Approach 1 — typed field for structured outputs / tool calling

If a UUID flows through a Pydantic model you hand to the SDK's structured
output or tool-calling helpers (`client.beta.chat.completions.parse(...,
response_format=YourModel)`, `client.responses.parse(...)`, function-calling
tool schemas, etc.), use `WordsUUID` as the field's type. Requires the
`openai` extra (`pip install uuid4-words[openai]`, which pulls in
`pydantic>=2`):

```python
from pydantic import BaseModel
from uuid_words.integrations.openai import WordsUUID

class LookupRecord(BaseModel):
    record_id: WordsUUID
    note: str

completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Look up record 173c6f57-..."}],
    response_format=LookupRecord,
)
record = completion.choices[0].message.parsed
record.record_id  # -> a real uuid.UUID, decoded from the word phrase
```

The JSON schema generated for `record_id` is a plain `string`, so the model
produces the word phrase directly (cheap on output tokens too); Pydantic
decodes it back to `uuid.UUID` on parse, and re-encodes to the word phrase
if you serialize the model again — e.g. to feed a tool result back into the
conversation. This is the preferred approach whenever it applies: the type
system, not a text scan, is what identifies the UUID.

### Approach 2 — explicit wrapper for freeform prompt/transcript text

If the UUID is just interpolated into ordinary prompt text (the main
token-savings use case — record IDs, trace IDs, etc. appearing in prompts or
long agent transcripts), there's no schema to attach a type to. Use the
`with_uuid_words` decorator to encode specific, named arguments of your own
prompt-building functions, wrapped in a delimiter so they can be found
unambiguously later:

```python
import uuid
from uuid_words.integrations.openai import with_uuid_words, extract_uuid_words

@with_uuid_words("record_id")
def build_prompt(record_id: uuid.UUID, question: str) -> str:
    return f"Regarding record {record_id}: {question}"

prompt = build_prompt(record_id=some_uuid, question="what changed?")
# "Regarding record ‹ability content shared ...›: what changed?"

# ... send `prompt` to the model as usual ...

# If the model echoes the phrase back in its response text:
extract_uuid_words(response_text)  # -> [UUID('...'), ...]
```

`with_uuid_words` only ever touches the keyword arguments you name — it
never scans the rest of the string — and `extract_uuid_words` only decodes
spans between the `‹ ›` delimiters it inserted, so there's no ambiguity
about what's a UUID versus an ordinary sentence that happens to reuse a few
wordlist words.

### Why not a global request/response rewriter?

You could intercept every request/response at the HTTP layer (e.g. via
`OpenAI(http_client=httpx.Client(transport=...))`) and blindly search-and-
replace UUID-shaped text. We don't ship that: it breaks down for streaming
responses (SSE chunks split words arbitrarily) and reintroduces the
false-positive risk of approach 2, just invisibly. Prefer explicit typing
(approach 1) or explicit delimiting (approach 2).

## Development

This package lives in the
[`shared-embeddings-space-experiments`](https://github.com/EdOlson-Morgan/shared-embeddings-space-experiments)
monorepo, as a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
member. From the repo root:

```bash
uv run --package uuid4-words pytest packages/uuid4-words/tests
```

To regenerate `src/uuid_words/wordlist.py` (requires network access to
`openaipublic.blob.core.windows.net` to verify single-token status against
`tiktoken`'s `o200k_base` vocab):

```bash
uv run --with wordfreq --with tiktoken scripts/build_uuid_wordlist.py
```

See [`PUBLISHING.md`](PUBLISHING.md) for the release process.

## License

MIT — see [`LICENSE`](LICENSE).
