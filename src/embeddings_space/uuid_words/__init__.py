"""Token-efficient word encoding for UUIDv4.

This is a thin re-export of the standalone, PyPI-published `uuid4-words`
package (`packages/uuid4-words/`, imports as `uuid_words`), kept for
existing in-repo imports. New code -- in this repo or elsewhere -- should
depend on `uuid_words` directly. See
docs/uuid-token-optimization-plan.md for the design.
"""

from uuid_words import decode_uuid4, encode_uuid4, encode_uuid4_slug

__all__ = ["decode_uuid4", "encode_uuid4", "encode_uuid4_slug"]
