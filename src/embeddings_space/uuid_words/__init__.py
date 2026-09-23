"""Token-efficient word encoding for UUIDv4.

See docs/uuid-token-optimization-plan.md for the design.
"""

from .codec import decode_uuid4, encode_uuid4, encode_uuid4_slug

__all__ = ["decode_uuid4", "encode_uuid4", "encode_uuid4_slug"]
