"""Reversible UUIDv4 <-> word-sequence encoding.

A UUIDv4 has 6 fixed bits (the version nibble and the top 2 bits of the
variant nibble) and 122 bits of actual randomness. This module strips the
fixed bits, packs the remaining 122 bits into a base-V integer where V is
the size of ``WORDLIST.WORDS``, and maps each base-V digit to a word.
Decoding reverses the process and reinserts the fixed version/variant bits.

See docs/uuid-token-optimization-plan.md for the design rationale.
"""

from __future__ import annotations

import re
import uuid

from .wordlist import WORDS

PAYLOAD_BITS = 122

# Bit positions (0 = least significant, matching uuid.UUID.int) that are
# fixed by the UUIDv4 spec rather than random: the version nibble (bits
# 76-79, i.e. hex digit 12) and the top 2 bits of the variant nibble
# (bits 62-63, i.e. the top 2 bits of hex digit 16).
_VERSION_BITS = frozenset(range(76, 80))
_VARIANT_BITS = frozenset((62, 63))
_FIXED_BITS = _VERSION_BITS | _VARIANT_BITS

# The 122 free (random) bit positions, most-significant first. This fixed
# ordering is what defines the payload integer's bit layout on both encode
# and decode.
_FREE_BITS: tuple[int, ...] = tuple(
    bit for bit in range(127, -1, -1) if bit not in _FIXED_BITS
)
assert len(_FREE_BITS) == PAYLOAD_BITS

_VOCAB_SIZE = len(WORDS)
if _VOCAB_SIZE & (_VOCAB_SIZE - 1) != 0:
    raise ValueError(f"wordlist size must be a power of two, got {_VOCAB_SIZE}")
_BITS_PER_WORD = _VOCAB_SIZE.bit_length() - 1
_WORD_COUNT = -(-PAYLOAD_BITS // _BITS_PER_WORD)  # ceil division
if _VOCAB_SIZE**_WORD_COUNT < 2**PAYLOAD_BITS:
    raise ValueError("wordlist is too small to cover all UUIDv4 payloads")

_WORD_INDEX: dict[str, int] = {word: index for index, word in enumerate(WORDS)}
if len(_WORD_INDEX) != len(WORDS):
    raise ValueError("wordlist contains duplicate words")

_SPLIT_RE = re.compile(r"[\s-]+")


def _uuid4_to_payload(value: uuid.UUID) -> int:
    if value.version != 4:
        raise ValueError(f"expected a UUIDv4, got version {value.version}: {value}")
    n = value.int
    payload = 0
    for bit in _FREE_BITS:
        payload = (payload << 1) | ((n >> bit) & 1)
    return payload


def _payload_to_uuid4(payload: int) -> uuid.UUID:
    if not 0 <= payload < 2**PAYLOAD_BITS:
        raise ValueError(f"payload out of range for a UUIDv4: {payload}")
    n = 0
    for index, bit in enumerate(_FREE_BITS):
        n |= ((payload >> (PAYLOAD_BITS - 1 - index)) & 1) << bit
    n |= 0b0100 << 76  # version = 4
    n |= 0b10 << 62  # variant = RFC 4122
    return uuid.UUID(int=n)


def _payload_to_words(payload: int) -> list[str]:
    digits: list[int] = []
    for _ in range(_WORD_COUNT):
        payload, digit = divmod(payload, _VOCAB_SIZE)
        digits.append(digit)
    digits.reverse()
    return [WORDS[digit] for digit in digits]


def _words_to_payload(words: list[str]) -> int:
    if len(words) != _WORD_COUNT:
        raise ValueError(f"expected {_WORD_COUNT} words, got {len(words)}: {words}")
    payload = 0
    for word in words:
        try:
            digit = _WORD_INDEX[word]
        except KeyError:
            raise ValueError(f"unrecognized word: {word!r}") from None
        payload = payload * _VOCAB_SIZE + digit
    return payload


def _coerce_uuid4(value: uuid.UUID | str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def encode_uuid4(value: uuid.UUID | str) -> str:
    """Encode a UUIDv4 as a space-separated phrase of ``_WORD_COUNT`` words."""
    payload = _uuid4_to_payload(_coerce_uuid4(value))
    return " ".join(_payload_to_words(payload))


def encode_uuid4_slug(value: uuid.UUID | str) -> str:
    """Encode a UUIDv4 as a hyphen-separated slug of ``_WORD_COUNT`` words.

    Convenient for filenames/URLs/identifiers, but hyphen-joining has not
    been verified to preserve one-token-per-word under o200k_base the way
    the space-joined form is expected to (see docs/uuid-token-optimization-plan.md);
    prefer ``encode_uuid4`` when token count matters.
    """
    payload = _uuid4_to_payload(_coerce_uuid4(value))
    return "-".join(_payload_to_words(payload))


def decode_uuid4(text: str) -> uuid.UUID:
    """Decode a word phrase or slug (as produced by ``encode_uuid4``/``encode_uuid4_slug``)."""
    words = [w for w in _SPLIT_RE.split(text.strip().lower()) if w]
    payload = _words_to_payload(words)
    return _payload_to_uuid4(payload)
