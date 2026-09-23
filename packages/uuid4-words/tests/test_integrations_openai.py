import uuid

import pytest

from uuid_words import encode_uuid4


class TestWithUuidWords:
    def test_encodes_named_kwarg_only(self):
        from uuid_words.integrations.openai import with_uuid_words

        @with_uuid_words("record_id")
        def build(record_id, question):
            return record_id, question

        u = uuid.uuid4()
        record_id, question = build(record_id=u, question="what changed?")
        assert question == "what changed?"
        assert record_id == f"‹{encode_uuid4(u)}›"

    def test_untouched_kwargs_pass_through(self):
        from uuid_words.integrations.openai import with_uuid_words

        @with_uuid_words("record_id")
        def build(record_id, other):
            return record_id, other

        record_id, other = build(record_id=None, other="unchanged")
        assert record_id is None
        assert other == "unchanged"


class TestExtractUuidWords:
    def test_round_trips_through_prompt_text(self):
        from uuid_words.integrations.openai import extract_uuid_words, with_uuid_words

        @with_uuid_words("record_id")
        def build_prompt(record_id, question):
            return f"Regarding record {record_id}: {question}"

        u = uuid.uuid4()
        prompt = build_prompt(record_id=u, question="what changed?")
        assert extract_uuid_words(prompt) == [u]

    def test_multiple_phrases_in_order(self):
        from uuid_words.integrations.openai import extract_uuid_words, with_uuid_words

        @with_uuid_words("a", "b")
        def build(a, b):
            return f"first {a} then {b}"

        u1, u2 = uuid.uuid4(), uuid.uuid4()
        text = build(a=u1, b=u2)
        assert extract_uuid_words(text) == [u1, u2]

    def test_no_phrases_returns_empty(self):
        from uuid_words.integrations.openai import extract_uuid_words

        assert extract_uuid_words("just an ordinary sentence with several words") == []

    def test_delimited_garbage_raises(self):
        from uuid_words.integrations.openai import extract_uuid_words

        with pytest.raises(ValueError):
            extract_uuid_words("‹not a real phrase›")


class TestWordsUuidType:
    def test_round_trips_via_pydantic_model(self):
        pydantic = pytest.importorskip("pydantic")
        from uuid_words.integrations.openai import WordsUUID

        class Record(pydantic.BaseModel):
            record_id: WordsUUID

        u = uuid.uuid4()
        # Model input as the model would actually produce it: the word phrase.
        parsed = Record.model_validate({"record_id": encode_uuid4(u)})
        assert parsed.record_id == u

        # Serializing back out produces the word phrase, not raw hex.
        assert parsed.model_dump()["record_id"] == encode_uuid4(u)

    def test_accepts_raw_hex_too(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        from uuid_words.integrations.openai import WordsUUID

        class Record(BaseModel):
            record_id: WordsUUID

        u = uuid.uuid4()
        parsed = Record.model_validate({"record_id": str(u)})
        assert parsed.record_id == u

    def test_json_schema_is_plain_string(self):
        pytest.importorskip("pydantic")
        from pydantic import BaseModel

        from uuid_words.integrations.openai import WordsUUID

        class Record(BaseModel):
            record_id: WordsUUID

        schema = Record.model_json_schema()
        assert schema["properties"]["record_id"]["type"] == "string"
