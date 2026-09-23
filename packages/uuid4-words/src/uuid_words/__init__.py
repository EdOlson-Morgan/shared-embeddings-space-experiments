"""Deterministic, reversible UUIDv4 <-> word-phrase encoding.

Encodes a UUIDv4 as a short sequence of plain English words instead of raw
hex, so it costs fewer LLM tokens in prompts and transcripts while staying
human-readable and losslessly reversible.

    >>> from uuid_words import encode_uuid4, decode_uuid4
    >>> encode_uuid4("173c6f57-2c5f-41a3-b898-f49eae04ffcc")
    'ability content shared prime queen think dude created fact frank bay wine'

See the README for the OpenAI SDK integration recipes
(``uuid_words.integrations.openai``) and the full design writeup linked
there.
"""

from .codec import decode_uuid4, encode_uuid4, encode_uuid4_slug

__all__ = ["decode_uuid4", "encode_uuid4", "encode_uuid4_slug"]

__version__ = "0.1.0"
