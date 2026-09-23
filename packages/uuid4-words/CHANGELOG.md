# Changelog

All notable changes to `uuid4-words` are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project uses [SemVer](https://semver.org/).

## [0.1.0] - Unreleased

Initial release.

- `encode_uuid4` / `decode_uuid4`: reversible UUIDv4 <-> space-joined word
  phrase encoding, verified single-token per word under `o200k_base`.
- `encode_uuid4_slug`: hyphen-joined variant for identifiers/URLs (lower
  token savings — see README).
- `uuid_words.integrations.openai`: `WordsUUID` pydantic field type, and
  `with_uuid_words` / `extract_uuid_words` for freeform prompt text.
