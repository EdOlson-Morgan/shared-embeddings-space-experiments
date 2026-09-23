"""Two independent ways to use ``uuid_words`` with the OpenAI Python SDK.

The ``openai`` SDK has no middleware/plugin hook for rewriting message or
tool-call content, so neither approach here patches the client. Pick
whichever matches where your UUIDs actually appear -- do not try to
auto-detect word-phrases in arbitrary freeform text (a run of common
English words is not a safe fingerprint for "this was an encoded UUID").

1. ``WordsUUID`` -- a pydantic field type for UUIDs that travel through
   structured outputs or tool-call arguments (i.e. anywhere the OpenAI SDK
   builds a JSON schema from a pydantic model and parses the model's JSON
   back into one, via ``.parse()``/``.beta.chat.completions.parse()``).
   The encode/decode is automatic on every round trip -- no text scanning,
   because the field is explicitly typed.

2. ``with_uuid_words`` / ``extract_uuid_words`` -- for UUIDs embedded in
   freeform prompt or transcript text. Since there's no schema to hang a
   type on, the word-phrase is wrapped in an explicit delimiter when it's
   interpolated into a message string, so it can be found unambiguously
   later with a regex rather than guessed at.
"""

from __future__ import annotations

import functools
import re
import uuid
from typing import Annotated, Any, Callable, TypeVar

from ..codec import decode_uuid4, encode_uuid4

__all__ = [
    "WordsUUID",
    "with_uuid_words",
    "extract_uuid_words",
    "UUID_WORDS_PATTERN",
]

_F = TypeVar("_F", bound=Callable[..., Any])


# --- Approach 1: typed field for structured outputs / tool calling -------

try:
    from pydantic import BeforeValidator, PlainSerializer
except ImportError:  # pragma: no cover - exercised via the `openai` extra
    WordsUUID = None
else:

    def _decode_words_or_hex(value: Any) -> Any:
        # Accept either form on input so the type also tolerates a model
        # (or a caller) that emits a plain hex UUID string instead of the
        # word phrase.
        if isinstance(value, str) and " " in value.strip():
            return decode_uuid4(value)
        return value

    WordsUUID = Annotated[
        uuid.UUID,
        BeforeValidator(_decode_words_or_hex),
        PlainSerializer(encode_uuid4, return_type=str),
    ]
    """``uuid.UUID``, but serializes to (and parses from) the token-efficient
    word phrase rather than the raw hex string.

    Use it as a field's type in any pydantic model you pass to the OpenAI
    SDK's structured-output / tool-calling helpers::

        from pydantic import BaseModel
        from uuid_words.integrations.openai import WordsUUID

        class LookupRecord(BaseModel):
            record_id: WordsUUID
            note: str

        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[...],
            response_format=LookupRecord,
        )
        completion.choices[0].message.parsed.record_id  # -> uuid.UUID

    The JSON schema generated for this field is a plain ``string`` (with no
    format hint), so the model produces the word phrase directly; pydantic
    decodes it back to a real ``uuid.UUID`` on parse, and re-encodes to the
    word phrase if you serialize the model again (e.g. to feed a follow-up
    tool result back to the API).
    """


# --- Approach 2: explicit wrapper for freeform prompt/transcript text ----

_DELIM_OPEN = "‹"  # single left-pointing angle quote: unlikely to
_DELIM_CLOSE = "›"  # collide with real prompt content or Markdown.

UUID_WORDS_PATTERN = re.compile(rf"{_DELIM_OPEN}([a-z]+(?: [a-z]+)*){_DELIM_CLOSE}")
"""Matches a delimited word phrase produced by ``with_uuid_words``."""


def with_uuid_words(*uuid_kwargs: str) -> Callable[[_F], _F]:
    """Decorator factory: encode the named keyword arguments before the call.

    Wrap a function that builds prompt text (a message-formatting helper, an
    f-string template function, etc.). Each argument named in
    ``uuid_kwargs`` is read as a ``uuid.UUID | str`` and replaced with its
    word-phrase encoding, wrapped in a delimiter, before the wrapped
    function runs::

        @with_uuid_words("record_id")
        def build_prompt(record_id: uuid.UUID, question: str) -> str:
            return f"Regarding record {record_id}: {question}"

        build_prompt(record_id=some_uuid, question="what changed?")
        # "Regarding record ‹ability content shared›: what changed?"

    Decode any phrases the model echoes back (or that appear in your own
    constructed prompt) with ``extract_uuid_words``.
    """

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for name in uuid_kwargs:
                value = kwargs.get(name)
                if value is not None:
                    kwargs[name] = f"{_DELIM_OPEN}{encode_uuid4(value)}{_DELIM_CLOSE}"
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def extract_uuid_words(text: str) -> list[uuid.UUID]:
    """Find and decode every delimited word phrase in ``text``.

    Pairs with ``with_uuid_words``: decodes exactly the phrases your own
    code encoded and delimited, in the order they appear. Raises
    ``ValueError`` (from ``decode_uuid4``) if a delimited span isn't a
    valid word phrase, rather than silently skipping it.
    """
    return [decode_uuid4(match.group(1)) for match in UUID_WORDS_PATTERN.finditer(text)]
