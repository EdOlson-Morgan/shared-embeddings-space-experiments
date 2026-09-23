"""Re-export of `uuid_words.codec` (see `packages/uuid4-words/`).

Kept so existing in-repo imports (including the private names used by
tests/test_uuid_words.py) keep working; `uuid_words.codec` is the source
of truth.
"""

from uuid_words.codec import (
    PAYLOAD_BITS,
    _VOCAB_SIZE,
    _WORD_COUNT,
    _coerce_uuid4,
    _payload_to_uuid4,
    _payload_to_words,
    _uuid4_to_payload,
    _words_to_payload,
    decode_uuid4,
    encode_uuid4,
    encode_uuid4_slug,
)

__all__ = [
    "PAYLOAD_BITS",
    "decode_uuid4",
    "encode_uuid4",
    "encode_uuid4_slug",
]
