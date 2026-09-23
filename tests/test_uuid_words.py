import random
import uuid

import pytest

from embeddings_space.uuid_words import decode_uuid4, encode_uuid4, encode_uuid4_slug
from embeddings_space.uuid_words.codec import (
    PAYLOAD_BITS,
    _VOCAB_SIZE,
    _WORD_COUNT,
    _payload_to_uuid4,
)
from embeddings_space.uuid_words.wordlist import WORDS


def _random_uuid4s(n: int, seed: int = 0) -> list[uuid.UUID]:
    rng = random.Random(seed)
    return [uuid.UUID(int=rng.getrandbits(128), version=4) for _ in range(n)]


class TestRoundTrip:
    def test_phrase_round_trip(self):
        for u in _random_uuid4s(2000):
            assert decode_uuid4(encode_uuid4(u)) == u

    def test_slug_round_trip(self):
        for u in _random_uuid4s(500, seed=1):
            assert decode_uuid4(encode_uuid4_slug(u)) == u

    def test_accepts_string_input(self):
        u = uuid.uuid4()
        assert encode_uuid4(str(u)) == encode_uuid4(u)

    def test_decode_is_case_and_whitespace_insensitive(self):
        u = uuid.uuid4()
        phrase = encode_uuid4(u)
        assert decode_uuid4(phrase.upper()) == u
        assert decode_uuid4("  " + phrase + "  ") == u
        assert decode_uuid4(phrase.replace(" ", "  ")) == u

    def test_edge_payloads(self):
        for payload in (0, 2**PAYLOAD_BITS - 1):
            u = _payload_to_uuid4(payload)
            assert u.version == 4
            assert decode_uuid4(encode_uuid4(u)) == u


class TestEncodingShape:
    def test_word_count(self):
        u = uuid.uuid4()
        assert len(encode_uuid4(u).split(" ")) == _WORD_COUNT

    def test_only_uses_wordlist_words(self):
        u = uuid.uuid4()
        assert set(encode_uuid4(u).split(" ")) <= set(WORDS)


class TestErrors:
    def test_encode_rejects_non_v4(self):
        v1 = uuid.uuid1()
        with pytest.raises(ValueError):
            encode_uuid4(v1)

    def test_decode_rejects_wrong_word_count(self):
        with pytest.raises(ValueError):
            decode_uuid4(" ".join(WORDS[:3]))

    def test_decode_rejects_unknown_word(self):
        words = list(WORDS[:_WORD_COUNT])
        words[0] = "definitelynotarealword"
        with pytest.raises(ValueError):
            decode_uuid4(" ".join(words))


class TestWordlist:
    def test_size_is_power_of_two(self):
        assert _VOCAB_SIZE & (_VOCAB_SIZE - 1) == 0

    def test_no_duplicates(self):
        assert len(WORDS) == len(set(WORDS))

    def test_all_lowercase_ascii_alpha(self):
        for word in WORDS:
            assert word.isascii() and word.isalpha() and word == word.lower()


class TestTokenSavings:
    """Gated on tiktoken + network access to the o200k_base vocab file.

    See docs/uuid-token-optimization-plan.md -- this sandbox's egress
    policy blocks openaipublic.blob.core.windows.net, so these tests are
    skipped here rather than failed. Run them from an environment that can
    reach that host to get real numbers.
    """

    @pytest.fixture(scope="class")
    def encoder(self):
        tiktoken = pytest.importorskip("tiktoken")
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception as exc:
            pytest.skip(f"o200k_base vocab unavailable: {exc!r}")

    def test_wordlist_is_single_token(self, encoder):
        multi_token = [w for w in WORDS if len(encoder.encode(" " + w)) != 1]
        assert not multi_token, f"{len(multi_token)} words are not single-token: {multi_token[:20]}"

    def test_phrase_uses_fewer_tokens_than_raw_uuid(self, encoder):
        uuids = _random_uuid4s(50, seed=2)
        raw_tokens = [len(encoder.encode(str(u))) for u in uuids]
        encoded_tokens = [len(encoder.encode(encode_uuid4(u))) for u in uuids]
        avg_raw = sum(raw_tokens) / len(raw_tokens)
        avg_encoded = sum(encoded_tokens) / len(encoded_tokens)
        print(f"avg raw tokens: {avg_raw:.1f}, avg encoded tokens: {avg_encoded:.1f}")
        assert avg_encoded < avg_raw
